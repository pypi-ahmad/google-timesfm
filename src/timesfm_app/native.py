"""Workspace-owned Windows services, with no containers or global PATH changes."""

from __future__ import annotations

import argparse
import ctypes
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / ".native"
LOGS = NATIVE / "logs"
PG = NATIVE / "postgresql" / "pgsql" / "bin"
PGDATA = NATIVE / "postgres-data"
STATE = NATIVE / "processes.json"
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(args, **kwargs):
  return subprocess.run(
    [str(arg) for arg in args], cwd=ROOT, creationflags=HIDDEN, check=True, **kwargs
  )


def download(url: str, destination: Path, sha256: str | None = None):
  destination.parent.mkdir(parents=True, exist_ok=True)
  if not destination.exists():
    temporary = destination.with_suffix(".part")
    with (
      urllib.request.urlopen(url, timeout=60) as response,
      temporary.open("wb") as out,
    ):
      shutil.copyfileobj(response, out)
    temporary.replace(destination)
  if sha256:
    digest = hashlib.sha256()
    with destination.open("rb") as handle:
      for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    if digest.hexdigest() != sha256:
      raise RuntimeError(
        "Downloaded package checksum does not match the pinned release."
      )


def bootstrap():
  """Extract vendor archives; never invoke an installer or write system settings."""
  if os.name != "nt":
    raise RuntimeError(
      "Native bootstrap targets Windows. Configure services manually elsewhere."
    )
  LOGS.mkdir(parents=True, exist_ok=True)
  if not (PG / "initdb.exe").exists():
    archive = NATIVE / "downloads/postgresql.zip"
    download(
      "https://get.enterprisedb.com/postgresql/postgresql-18.6-3-windows-x64-binaries.zip",
      archive,
    )
    destination = NATIVE / "postgresql"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
      for item in bundle.infolist():
        if (
          not (destination / item.filename)
          .resolve()
          .is_relative_to(destination.resolve())
        ):
          raise RuntimeError("Unsafe path in PostgreSQL archive.")
      bundle.extractall(destination)
  memurai = NATIVE / "memurai/memurai.exe"
  if not memurai.exists() or not (memurai.parent / "memurai-services.dll").exists():
    import olefile

    package = NATIVE / "downloads/memurai.msi"
    download(
      "https://dist.memurai.com/releases/Memurai-Developer/4.1.2/Memurai-Developer-v4.1.2.msi",
      package,
      "11d009c8e93912e1899b5263d0e7d78907903913e1672b30020612ada6d27446",
    )
    cabinet = NATIVE / "downloads/memurai.cab"
    with olefile.OleFileIO(package) as archive:
      for entry in archive.listdir():
        payload = archive.openstream(entry).read()
        if payload.startswith(b"MSCF"):
          cabinet.write_bytes(payload)
          break
      else:
        raise RuntimeError("No cabinet found in the pinned Memurai archive.")
    memurai.parent.mkdir(parents=True, exist_ok=True)
    run(["expand.exe", "-F:*", cabinet, memurai.parent], stdout=subprocess.DEVNULL)
    for source, target in [
      ("file_memurai.exe", "memurai.exe"),
      ("file_memurai_services.dll", "memurai-services.dll"),
      ("libcrypto_3_x64.dll", "libcrypto-3-x64.dll"),
      ("libssl_3_x64.dll", "libssl-3-x64.dll"),
    ]:
      shutil.copy2(memurai.parent / source, memurai.parent / target)
  if not (PGDATA / "PG_VERSION").exists():
    run(
      [
        PG / "initdb.exe",
        "-D",
        PGDATA,
        "-U",
        "timesfm",
        "--auth-host=sspi",
        "--encoding=UTF8",
        "--locale=C",
      ],
      stdout=subprocess.DEVNULL,
    )
    # SSPI maps this Windows account to the private database owner. No password
    # is generated, stored, printed, or added to environment variables.
    account = getpass.getuser().replace('"', '""')
    (PGDATA / "pg_ident.conf").write_text(
      f'timesfm_local "{account}" timesfm\n', encoding="utf-8"
    )
    (PGDATA / "pg_hba.conf").write_text(
      "host all timesfm 127.0.0.1/32 sspi map=timesfm_local include_realm=0\n",
      encoding="utf-8",
    )
  (NATIVE / "memurai.conf").write_text(
    "bind 127.0.0.1\nport 56379\nprotected-mode yes\nappendonly yes\n"
    + f'dir "{(NATIVE / "memurai").as_posix()}"\n',
    encoding="utf-8",
  )
  print("Native PostgreSQL and Memurai archives are ready.")


def creation_filetime(pid):
  """Read exact Windows process birth identity without reading its environment."""
  if os.name != "nt":
    return None
  from ctypes import wintypes

  kernel = ctypes.WinDLL("kernel32", use_last_error=True)
  kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
  kernel.OpenProcess.restype = wintypes.HANDLE
  kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
    ctypes.POINTER(wintypes.FILETIME)
  ] * 4
  kernel.GetProcessTimes.restype = wintypes.BOOL
  kernel.CloseHandle.argtypes = [wintypes.HANDLE]
  handle = kernel.OpenProcess(0x1000, False, pid)
  if not handle:
    error = ctypes.get_last_error()
    if error in {87, 1168}:
      raise psutil.NoSuchProcess(pid)
    if error == 5:
      raise psutil.AccessDenied(pid)
    raise ctypes.WinError(error)
  try:
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
      raise ctypes.WinError(ctypes.get_last_error())
    return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
  finally:
    kernel.CloseHandle(handle)


def identity(process):
  process = psutil.Process(process.pid)
  return {
    "pid": process.pid,
    "created": process.create_time(),
    "creation_filetime": creation_filetime(process.pid),
  }


def existing(value):
  try:
    process = psutil.Process(value["pid"])
    created = process.create_time()
    # Windows can retain a terminated process handle that denies OpenProcess
    # while psutil's system process query already proves the PID is gone.
    if value.get("created") is not None and created != value["created"]:
      return None
    if os.name == "nt" and value.get("creation_filetime") is not None:
      try:
        matches = creation_filetime(process.pid) == value["creation_filetime"]
      except psutil.AccessDenied:
        # Recheck a possible exit between the birth query and opening a handle.
        current = psutil.Process(process.pid).create_time()
        if current != created or not psutil.pid_exists(process.pid):
          return None
        raise  # Unverifiable is distinct from dead; cancellation cannot complete.
    else:
      # Legacy state and non-Windows tests use psutil's repeatable birth value.
      matches = process.create_time() == value["created"]
    return process if matches else None
  except (psutil.NoSuchProcess, KeyError, TypeError, ValueError):
    return None


def save_state(state):
  """Replace the ownership journal atomically, preserving it across crashes."""
  temporary = STATE.with_name(f".{STATE.name}.{uuid.uuid4().hex}.tmp")
  try:
    temporary.write_text(json.dumps(state), encoding="utf-8")
    os.replace(temporary, STATE)
  finally:
    temporary.unlink(missing_ok=True)


def launch(name, command, session_token=None):
  LOGS.mkdir(parents=True, exist_ok=True)
  with (LOGS / f"{name}.log").open("ab") as log:
    environment = None
    if session_token:
      environment = {**os.environ, "TIMESFM_SUPERVISOR_SESSION": session_token}
    return subprocess.Popen(
      [str(arg) for arg in command],
      cwd=ROOT,
      stdout=log,
      stderr=log,
      creationflags=HIDDEN,
      env=environment,
    )


def infrastructure():
  if not (PG / "pg_ctl.exe").exists():
    raise RuntimeError("Run .\\dev.ps1 setup first.")
  status = subprocess.run(
    [str(PG / "pg_ctl.exe"), "-D", str(PGDATA), "status"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    creationflags=HIDDEN,
    check=False,
  )
  if status.returncode:
    run(
      [
        PG / "pg_ctl.exe",
        "-D",
        PGDATA,
        "-l",
        LOGS / "postgres.log",
        "-o",
        "-h 127.0.0.1 -p 55432",
        "-w",
        "start",
      ],
      stdout=subprocess.DEVNULL,
    )
  import psycopg
  from psycopg import sql

  with psycopg.connect(
    "host=127.0.0.1 port=55432 user=timesfm dbname=postgres", autocommit=True
  ) as connection:
    if not connection.execute(
      "SELECT 1 FROM pg_database WHERE datname = %s", ["timesfm"]
    ).fetchone():
      connection.execute(
        sql.SQL("CREATE DATABASE {} ").format(sql.Identifier("timesfm"))
      )
  from redis import Redis

  redis = Redis.from_url("redis://127.0.0.1:56379/0", socket_connect_timeout=1)
  try:
    redis.ping()
    memurai = None
  except Exception:  # noqa: BLE001 - report broker startup failures without credentials.
    memurai = launch(
      "memurai", [NATIVE / "memurai/memurai.exe", NATIVE / "memurai.conf"]
    )
    for _ in range(30):
      try:
        redis.ping()
        break
      except Exception:  # noqa: BLE001 - bounded readiness check for the broker.
        time.sleep(0.5)
    else:
      raise RuntimeError(
        "Memurai did not start. See .native/logs/memurai.log."
      ) from None
  run([sys.executable, "-m", "alembic", "upgrade", "head"], stdout=subprocess.DEVNULL)
  return memurai


def terminate_owned(process):
  """Terminate only a process with the recorded creation time, then await exit."""
  current = existing(process)
  if current is None:
    return
  pending, stopped = [current], []
  try:
    while pending:
      child = pending.pop()
      try:
        # Freeze each parent before discovering its children so a launcher cannot
        # replace a killed worker or spawn an unobserved child during shutdown.
        child.suspend()
        stopped.append(child)
        pending.extend(child.children())
      except psutil.NoSuchProcess:
        continue
    for child in stopped:
      try:
        child.kill()
      except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs(stopped, timeout=15)
    if alive:
      raise RuntimeError("An owned process did not exit; replacement was refused.")
  finally:
    for child in stopped:
      try:
        child.resume()
      except psutil.NoSuchProcess:
        pass


def owned_worker_records(store, session_token):
  """Resolve inherited session identity plus process birth time, never PID alone."""
  found = []
  for record in store.list_records("worker", None):
    payload = record["payload"]
    if not session_token or payload.get("supervisor_session") != session_token:
      continue
    try:
      process = existing(worker_identity(payload))
      if process is not None:
        found.append((payload, identity(process)))
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
      continue
  return found


def worker_identity(payload):
  filetime = payload.get("creation_filetime")
  if os.name == "nt":
    if not isinstance(filetime, str) or not filetime.isdecimal():
      raise ValueError("Worker birth identity is unavailable.")
    # Only compare psutil timestamps if they were recorded through that API.
    # Converting FILETIME to a float can round differently from psutil.
    created = payload.get("process_created")
  else:
    created = payload["process_created"]
  return {"pid": payload["pid"], "created": created, "creation_filetime": filetime}


def stop_group(store, state, name):
  """Stop the launcher before recorded descendants, including orphaned workers."""
  confirmed = True
  identities = []
  if name in state:
    identities.append(state[name])
  identities.extend(state.get("descendants", {}).get(name, {}).values())
  for child in identities:
    try:
      terminate_owned(child)
    except psutil.AccessDenied:
      confirmed = False
  for record in store.list_records("worker", None):
    payload = record["payload"]
    if (
      state.get("session")
      and payload.get("supervisor_session") == state["session"]
      and payload.get("queue") == name
    ):
      try:
        terminate_owned(worker_identity(payload))
      except (psutil.AccessDenied, KeyError, TypeError, ValueError):
        confirmed = False
  return confirmed


def confirmed_gone(identity):
  try:
    return existing(identity) is None
  except psutil.AccessDenied:
    return False


def stop_cancelled_job(store, state, job_id, attempt, queue):
  """Called while the job row is locked; require its actual worker identity."""
  matches = []
  for record in store.list_records("worker", None):
    payload = record["payload"]
    if (
      payload.get("supervisor_session") == state.get("session")
      and state.get("session")
      and payload.get("queue") == queue
      and payload.get("current_job_id") == job_id
      and payload.get("current_attempt") == attempt
    ):
      try:
        matches.append(worker_identity(payload))
      except (KeyError, TypeError, ValueError):
        return False
  if not matches:
    return False
  if not all(confirmed_gone(child) for child in matches) and not stop_group(
    store, state, queue
  ):
    return False
  return all(confirmed_gone(child) for child in matches)


def cleanup_previous(store, previous):
  """Retire a dead supervisor's recorded processes before replacing its state."""
  if not previous:
    return
  if existing(previous.get("supervisor")) is not None:
    raise RuntimeError("The previous workbench supervisor is still running.")
  stopped = [
    stop_group(store, previous, name)
    for name in ("web", "cpu", "gpu", "dispatcher", "api")
  ]
  # Dead worker records retain the job/attempt needed to acknowledge cancellation
  # even when a replacement launcher is already alive.
  for record in store.list_records("worker", None):
    payload = record["payload"]
    if (
      previous.get("session")
      and payload.get("supervisor_session") == previous["session"]
      and payload.get("current_job_id")
      and payload.get("current_attempt") is not None
    ):
      store.confirm_cancel_after_exit(
        payload["current_job_id"],
        payload["current_attempt"],
        lambda item=payload: confirmed_gone(worker_identity(item)),
      )
  if not all(stopped):
    raise RuntimeError(
      "Some recorded processes could not be verified or stopped; ownership state was preserved."
    )


def supervise(dev=False):
  import msvcrt

  from .config import get_settings
  from .store import Store

  NATIVE.mkdir(exist_ok=True)
  lock = (NATIVE / "launcher.lock").open("a+b")
  lock.seek(0)
  lock.write(b"0")
  lock.flush()
  lock.seek(0)
  try:
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
  except OSError:
    raise RuntimeError("The workbench supervisor is already running.") from None
  config = get_settings()
  store = Store(config.database_url)
  memurai = infrastructure()
  previous = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
  cleanup_previous(store, previous)
  worker_command = [
    sys.executable,
    "-m",
    "dramatiq",
    "timesfm_app.worker",
    "--use-spawn",
    "--processes",
    "1",
    "--threads",
    "1",
    "--queues",
  ]
  next_cli = ROOT / "web/node_modules/next/dist/bin/next"
  commands = {
    "api": [sys.executable, "-m", "timesfm_app.api"],
    "dispatcher": [sys.executable, "-m", "timesfm_app.jobs"],
    "gpu": [*worker_command, "gpu"],
    "cpu": [*worker_command, "cpu"],
    "web": [
      shutil.which("node") or "node",
      next_cli,
      "dev" if dev else "start",
      str(ROOT / "web"),
      "--hostname",
      "127.0.0.1",
      "--port",
      str(config.frontend_port),
    ],
  }
  session_token = uuid.uuid4().hex
  processes = {}
  state = {
    "supervisor": identity(psutil.Process()),
    "session": session_token,
    "descendants": {name: {} for name in commands},
  }
  if memurai:
    state["memurai"] = identity(memurai)
  elif existing(previous.get("memurai")) is not None:
    state["memurai"] = previous["memurai"]
  save_state(state)
  stopping: dict[str, float] = {}
  descendants = state["descendants"]

  try:
    for name, command in commands.items():
      processes[name] = launch(name, command, session_token)
      state[name] = identity(processes[name])
      save_state(state)
    while not (NATIVE / "stop.request").exists():
      for name, process in list(processes.items()):
        parent = existing(state[name])
        changed = False
        if parent:
          for child in parent.children(recursive=True):
            try:
              child_identity = identity(child)
              key = str(child.pid)
              if descendants[name].get(key) != child_identity:
                descendants[name][key] = child_identity
                changed = True
            except psutil.NoSuchProcess:
              pass
        if changed:
          save_state(state)
        if process.poll() is not None:
          stop_group(store, state, name)
          descendants[name] = {}
          processes[name] = launch(name, commands[name], session_token)
          state[name] = identity(processes[name])
          save_state(state)
      try:
        workspaces = store.list_records("workspace", None)
        for workspace in workspaces:
          for job in store.list_jobs(workspace["id"]):
            if job["status"] != "cancelling":
              stopping.pop(job["id"], None)
              continue
            started = stopping.setdefault(job["id"], time.monotonic())
            if time.monotonic() - started < config.cancellation_grace_seconds:
              continue
            name = "cpu" if job["kind"] in {"assessment", "model_check"} else "gpu"

            store.confirm_cancel_after_exit(
              job["id"],
              job["attempt"],
              lambda current=job, group=name: stop_cancelled_job(
                store, state, current["id"], current["attempt"], group
              ),
            )
            stopping.pop(job["id"], None)
      except Exception as exc:  # noqa: BLE001 - keep recovery alive without logging secrets.
        print(
          f"Supervisor reconciliation unavailable ({type(exc).__name__}).", flush=True
        )
      time.sleep(1)
  finally:
    stopped = [stop_group(store, state, name) for name in reversed(list(processes))]
    if "memurai" in state:
      terminate_owned(state["memurai"])
    store.close()
    lock.close()
    if not all(stopped):
      raise RuntimeError(
        "Some recorded processes could not be verified or stopped; ownership state was preserved."
      )
    (NATIVE / "stop.request").unlink(missing_ok=True)
    STATE.unlink(missing_ok=True)


def http_ready(url, *, api=False):
  opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(url, timeout=2) as response:
    if response.status != 200:
      raise RuntimeError("HTTP service is not ready.")
    if api and json.load(response).get("status") != "ok":
      raise RuntimeError("API health is degraded.")


def doctor():
  from redis import Redis

  from .config import get_settings
  from .store import Store

  config = get_settings()
  results = {
    "python": sys.version.split()[0],
    "native_postgresql": (PG / "pg_ctl.exe").exists(),
    "native_memurai": (NATIVE / "memurai/memurai.exe").exists(),
  }
  for name, action in [
    ("database", lambda: Store(config.database_url).list_records("workspace", None)),
    (
      "queue",
      lambda: Redis.from_url(config.redis_url, socket_connect_timeout=2).ping(),
    ),
    (
      "api",
      lambda: http_ready(f"http://127.0.0.1:{config.api_port}/api/v1/health", api=True),
    ),
    ("frontend", lambda: http_ready(f"http://127.0.0.1:{config.frontend_port}/")),
  ]:
    try:
      action()
      results[name] = "ready"
    except Exception as exc:  # noqa: BLE001 - diagnostic output contains exception types only.
      results[name] = f"unavailable ({type(exc).__name__})"
  print(json.dumps(results, indent=2))
  return all(
    results[name] == "ready" for name in ("database", "queue", "api", "frontend")
  )


def main():
  """Run the Windows-native setup, supervisor, health, or stop command."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "command",
    choices=["setup", "start", "stop", "doctor", "supervise", "infrastructure"],
  )
  parser.add_argument("--dev", action="store_true")
  args = parser.parse_args()
  if args.command == "setup":
    bootstrap()
  elif args.command == "infrastructure":
    infrastructure()
  elif args.command == "supervise":
    supervise(args.dev)
  elif args.command == "doctor":
    if not doctor():
      raise SystemExit(1)
  elif args.command == "stop":
    if STATE.exists():
      state = json.loads(STATE.read_text())
      (NATIVE / "stop.request").touch()
      supervisor = existing(state["supervisor"])
      if supervisor:
        supervisor.wait(timeout=60)
        if STATE.exists():
          raise RuntimeError(
            "The supervisor exited before cleanup completed. Ownership state remains; run stop again to reconcile it."
          )
      else:
        from .config import get_settings
        from .store import Store

        store = Store(get_settings().database_url)
        try:
          cleanup_previous(store, state)
          if "memurai" in state:
            terminate_owned(state["memurai"])
          STATE.unlink(missing_ok=True)
          (NATIVE / "stop.request").unlink(missing_ok=True)
        finally:
          store.close()
    print("Workbench processes stopped. PostgreSQL remains available for saved data.")
  else:
    if STATE.exists() and existing(json.loads(STATE.read_text())["supervisor"]):
      print("The workbench is already running.")
      return
    (NATIVE / "stop.request").unlink(missing_ok=True)
    launch(
      "supervisor",
      [
        sys.executable,
        "-m",
        "timesfm_app.native",
        "supervise",
        *(["--dev"] if args.dev else []),
      ],
    )
    print("Starting workbench: http://127.0.0.1:3000 — logs in .native/logs")


if __name__ == "__main__":
  main()
