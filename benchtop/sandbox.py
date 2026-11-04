import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{tmp}") (subpath "/dev"))
{deny_reads}
"""


def default_deny_reads() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return [root / "envs", root / ".state", root / "docs", root / "artifacts"]


def sandbox_mode(requested: str = "auto") -> str:
    if requested not in ("auto", "macos"):
        raise RuntimeError("unsupported sandbox mode; unisolated execution is disabled")
    if sys.platform != "darwin" or not shutil.which("sandbox-exec"):
        raise RuntimeError("live execution requires the supported macOS sandbox-exec backend")
    return "macos"


class Worker:
    def __init__(self, lab_path: Path, mode: str, log_path: Path | None, deny_reads: list[Path], expected_sha: str):
        self.lab_path = Path(lab_path).resolve()
        self.mode = mode
        self.log_path = log_path
        self.deny_reads = deny_reads
        self.expected_sha = expected_sha
        self.proc = None
        self.tmp = None

    async def start(self, timeout: float) -> None:
        sandbox_mode(self.mode)
        self.tmp = tempfile.mkdtemp(prefix="benchtop-")
        tmp = os.path.realpath(self.tmp)
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": tmp,
            "TMPDIR": tmp,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
        }
        env.update({v: "1" for v in THREAD_VARS})
        # the lab runs from a private copy so results and docs can stay unreadable
        lab_copy = Path(tmp) / "lab.py"
        shutil.copyfile(self.lab_path, lab_copy)
        if hashlib.sha256(lab_copy.read_bytes()).hexdigest() != self.expected_sha:
            await self.stop()
            raise RuntimeError("lab changed before sandbox worker startup")
        cmd = [sys.executable, "-I", "-m", "benchtop._worker", str(lab_copy)]
        if self.mode == "macos":
            deny = "\n".join(f'(deny file-read* (subpath "{os.path.realpath(d)}"))' for d in self.deny_reads)
            cmd = ["sandbox-exec", "-p", PROFILE.format(tmp=tmp, deny_reads=deny)] + cmd
        stderr = open(self.log_path, "ab") if self.log_path else asyncio.subprocess.DEVNULL
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            cwd=tmp,
            env=env,
            limit=1 << 22,
        )
        if self.log_path:
            stderr.close()
        line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        hello = json.loads(line or b'{"ready": false, "error": "worker exited during startup"}')
        if not hello.get("ready"):
            await self.stop()
            raise RuntimeError(f"lab failed to load: {hello.get('error')}\n{hello.get('trace', '')}")

    async def run(self, config: dict, seed: int, timeout: float) -> dict:
        self.proc.stdin.write((json.dumps({"config": config, "seed": seed}) + "\n").encode())
        await self.proc.stdin.drain()
        try:
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        except TimeoutError:
            await self.stop()
            return {"ok": False, "error": f"timeout after {timeout:.0f}s", "secs": timeout}
        if not line:
            code = await self.proc.wait()
            await self.stop()
            return {"ok": False, "error": f"worker crashed (exit {code})", "secs": 0.0}
        return json.loads(line)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def stop(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()
        self.proc = None
        if self.tmp:
            shutil.rmtree(self.tmp, ignore_errors=True)
            self.tmp = None


class Pool:
    def __init__(
        self,
        lab_path: Path,
        workers: int = 4,
        timeout: float = 60.0,
        sandbox: str = "auto",
        log_dir: Path | None = None,
        startup_timeout: float = 120.0,
        deny_reads: list[Path] | None = None,
    ):
        self.lab_path = Path(lab_path).resolve()
        self.deny_reads = [self.lab_path.parent, *(default_deny_reads() if deny_reads is None else deny_reads)]
        self.size = workers
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.mode = sandbox_mode(sandbox)
        self.expected_sha = hashlib.sha256(self.lab_path.read_bytes()).hexdigest()
        self.log_dir = Path(log_dir) if log_dir else None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.workers: list[Worker] = []

    async def __aenter__(self) -> "Pool":
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.size):
            log = self.log_dir / f"worker-{i}.log" if self.log_dir else None
            self.workers.append(Worker(self.lab_path, self.mode, log, self.deny_reads, self.expected_sha))
        # fail fast if the lab can't even import
        try:
            await self.workers[0].start(self.startup_timeout)
        except BaseException:
            await self.__aexit__()
            raise
        self.tasks = [asyncio.create_task(self._loop(w)) for w in self.workers]
        return self

    async def __aexit__(self, *exc) -> None:
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for w in self.workers:
            await w.stop()

    async def _loop(self, worker: Worker) -> None:
        while True:
            config, seed, fut = await self.queue.get()
            try:
                if not worker.alive:
                    await worker.start(self.startup_timeout)
                result = await worker.run(config, seed, self.timeout)
            except Exception as e:
                result = {"ok": False, "error": f"{type(e).__name__}: {e}", "secs": 0.0}
            if not fut.done():
                fut.set_result(result)

    async def run(self, config: dict, seed: int) -> dict:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((config, seed, fut))
        return await fut


async def run_jobs(
    lab_path: Path,
    jobs: list[tuple[dict, int]],
    workers: int = 4,
    timeout: float = 60.0,
    sandbox: str = "auto",
    log_dir: Path | None = None,
    on_result=None,
) -> list[dict]:
    async with Pool(lab_path, workers, timeout, sandbox, log_dir) as pool:

        async def one(config, seed):
            result = await pool.run(config, seed)
            if on_result:
                on_result(config, seed, result)
            return result

        return await asyncio.gather(*(one(c, s) for c, s in jobs))
