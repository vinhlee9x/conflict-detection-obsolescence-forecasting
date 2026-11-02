"""Download the Public Jira Dataset dump and restore chosen Jira instances into a local MongoDB.

The Zenodo zip (about 5.8 GB) holds one gzip mongodump archive of the whole database.
Restoring everything takes about 60 GB, more than a Colab disk, so only the selected
collections are restored, streamed straight out of the zip without extracting it.
"""

import hashlib
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ZENODO_RECORD = "15719919"
ZIP_NAME = "2025-06-23 ThePublicJiraDataset.zip"
ZIP_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/2025-06-23%20ThePublicJiraDataset.zip?download=1"
ZIP_SIZE = 5_813_135_238
ZIP_MD5 = "02f85309d966092ea130ca0797aea795"
ARCHIVE_MEMBER = "ThePublicJiraDataset/3. DataDump/mongodump-JiraReposAnon.archive"
DATABASE = "JiraReposAnon"
MONGO_URI = "mongodb://localhost:27017"
MONGODB_SERIES = "8.0"


def md5sum(path: Path, chunk_size: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_streaming(command: list[str]) -> None:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    if process.wait() != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def download_dump(dest_dir: Path, url: str = ZIP_URL, size: int = ZIP_SIZE, md5: str | None = ZIP_MD5) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / ZIP_NAME
    if not (target.exists() and target.stat().st_size == size):
        command = ["curl", "-sSfL", "--retry", "10", "--retry-all-errors", "--retry-delay", "5", "-C", "-", "-o", str(target), url]
        process = subprocess.Popen(command)
        while process.poll() is None:
            done = target.stat().st_size if target.exists() else 0
            print(f"Đã tải {done / 1e9:.2f} / {size / 1e9:.2f} GB", flush=True)
            time.sleep(30)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    if target.stat().st_size != size:
        raise RuntimeError(f"{target} có {target.stat().st_size} byte, cần {size}. Chạy lại ô này để tải tiếp.")
    if md5 and md5sum(target) != md5:
        raise RuntimeError(f"MD5 của {target} không khớp. Xoá file rồi tải lại.")
    return target


def apt_source_line(codename: str, series: str = MONGODB_SERIES) -> str:
    return (
        f"deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-{series}.gpg ] "
        f"https://repo.mongodb.org/apt/ubuntu {codename}/mongodb-org/{series} multiverse"
    )


def os_codename(os_release: Path = Path("/etc/os-release")) -> str:
    for line in os_release.read_text().splitlines():
        if line.startswith("VERSION_CODENAME="):
            return line.split("=", 1)[1].strip('"')
    raise RuntimeError("Không đọc được VERSION_CODENAME, máy không phải Ubuntu?")


def install_mongodb(series: str = MONGODB_SERIES) -> None:
    if shutil.which("mongod") and shutil.which("mongorestore"):
        return
    keyring = f"/usr/share/keyrings/mongodb-server-{series}.gpg"
    script = f"""set -e
curl -fsSL https://www.mongodb.org/static/pgp/server-{series}.asc | gpg --dearmor --yes -o {keyring}
echo {shlex.quote(apt_source_line(os_codename(), series))} > /etc/apt/sources.list.d/mongodb-org-{series}.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mongodb-org-server mongodb-database-tools"""
    run_streaming(["bash", "-c", script])


def mongo_ready(uri: str = MONGO_URI, timeout_ms: int = 1000) -> bool:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    try:
        MongoClient(uri, serverSelectionTimeoutMS=timeout_ms).admin.command("ping")
        return True
    except PyMongoError:
        return False


def start_mongod(dbpath: Path, port: int = 27017, wait_seconds: int = 60) -> str:
    uri = f"mongodb://localhost:{port}"
    if mongo_ready(uri):
        return uri
    dbpath.mkdir(parents=True, exist_ok=True)
    command = ["mongod", "--dbpath", str(dbpath), "--port", str(port), "--bind_ip", "127.0.0.1",
               "--fork", "--logpath", str(dbpath / "mongod.log")]
    subprocess.run(command, check=True)
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if mongo_ready(uri):
            return uri
        time.sleep(1)
    raise RuntimeError(f"mongod không phản hồi sau {wait_seconds}s, xem {dbpath / 'mongod.log'}")


def restore_command(zip_path: Path, collections: list[str], database: str = DATABASE, uri: str = MONGO_URI) -> str:
    includes = " ".join(f"--nsInclude={shlex.quote(f'{database}.{name}')}" for name in collections)
    return (
        f"set -o pipefail; unzip -p {shlex.quote(str(zip_path))} {shlex.quote(ARCHIVE_MEMBER)} "
        f"| mongorestore --uri={shlex.quote(uri)} --gzip --archive {includes} --drop --numInsertionWorkersPerCollection=4"
    )


def restore_collections(zip_path: Path, collections: list[str], database: str = DATABASE, uri: str = MONGO_URI) -> dict[str, int]:
    missing = [name for name in collections if restored_count(name, database, uri) == 0]
    if missing:
        run_streaming(["bash", "-c", restore_command(zip_path, missing, database, uri)])
    return {name: restored_count(name, database, uri) for name in collections}


def restored_count(collection: str, database: str = DATABASE, uri: str = MONGO_URI) -> int:
    from pymongo import MongoClient

    return MongoClient(uri)[database][collection].estimated_document_count()


def project_overview(collection: str, database: str = DATABASE, uri: str = MONGO_URI):
    import pandas as pd
    from pymongo import MongoClient

    from src.datasets.jira_export import LABEL_LINK_TYPES

    label_prefix = "^(" + "|".join(LABEL_LINK_TYPES) + ")"
    pipeline = [
        {"$project": {
            "project": "$fields.project.key",
            "links": {"$size": {"$ifNull": ["$fields.issuelinks", []]}},
            "label_links": {"$size": {"$filter": {
                "input": {"$ifNull": ["$fields.issuelinks", []]},
                "cond": {"$regexMatch": {"input": {"$ifNull": ["$$this.type.name", ""]}, "regex": label_prefix, "options": "i"}},
            }}},
        }},
        {"$group": {"_id": "$project", "issues": {"$sum": 1}, "links": {"$sum": "$links"}, "label_links": {"$sum": "$label_links"}}},
    ]
    rows = list(MongoClient(uri)[database][collection].aggregate(pipeline, allowDiskUse=True))
    overview = pd.DataFrame(rows).rename(columns={"_id": "project"})
    if overview.empty:
        return pd.DataFrame(columns=["project", "issues", "links", "label_links"])
    return overview.sort_values("label_links", ascending=False).reset_index(drop=True)


def suggest_projects(overview, count: int = 3, max_issues: int = 5000, min_issues: int = 500) -> list[str]:
    candidates = overview[(overview["issues"] >= min_issues) & (overview["issues"] <= max_issues)]
    if candidates.empty:
        raise ValueError(f"Không có project nào có {min_issues}–{max_issues} issue, hãy nới khoảng hoặc đổi collection.")
    ranked = candidates.sort_values(["label_links", "issues"], ascending=False)
    return [str(key) for key in ranked["project"].head(count)]
