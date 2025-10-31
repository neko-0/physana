import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

from .file_metadata import FileMetaData


class FileSQLiteDB:
    def __init__(self, db_path="ntuple_metadata.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.handle_duplicates = True
        self.exist_ok = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def close(self):
        self.conn.close()

    def _create_tables(self):
        cur = self.conn.cursor()

        # File metadata
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS file_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                campaign TEXT NOT NULL,
                dataset_id INTEGER NOT NULL,
                e_tag TEXT NOT NULL,
                num_executed_files INTEGER NOT NULL,
                num_input_events INTEGER NOT NULL,
                reco_tree_entries INTEGER NOT NULL,
                particle_tree_entries INTEGER NOT NULL,
                num_trees INTEGER NOT NULL,
                ab_version INTEGER NOT NULL,
                tcpt_version INTEGER NOT NULL,
                vjj_version INTEGER NOT NULL,
                file_path TEXT NOT NULL UNIQUE
            )
        """
        )

        # CutBookkeeper
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cutbookkeeper (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                run_number INTEGER NOT NULL,
                syst TEXT NOT NULL,
                total_events REAL NOT NULL,
                sum_weights REAL NOT NULL,
                sum_weights_sq REAL NOT NULL,
                FOREIGN KEY (file_id) REFERENCES file_metadata(id)
            )
        """
        )

        # Campaign summary: per campaign, dataset, systematics
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_summary (
                campaign TEXT NOT NULL,
                dataset_id INTEGER NOT NULL,
                syst TEXT NOT NULL,
                num_executed_files INTEGER NOT NULL,
                num_input_events INTEGER NOT NULL,
                total_events REAL NOT NULL,
                sum_weights REAL NOT NULL,
                sum_weights_sq REAL NOT NULL,
                PRIMARY KEY (campaign, dataset_id, syst)
            )
        """
        )

        self.conn.commit()

    def insert_file(self, metadata: FileMetaData, commit: bool = True) -> int:
        """
        Insert file metadata and cutbookkeeper into database,
        and incrementally update the campaign_summary table.
        """
        cur = self.conn.cursor()

        # Check for duplication
        if self.handle_duplicates:
            # Removing existing file path
            if not self.remove_file(metadata.file_path, commit=commit):
                return -1

        # Insert file metadata
        insert_query = """
            INSERT INTO file_metadata (
                data_type, campaign, dataset_id, e_tag,
                num_executed_files, num_input_events,
                reco_tree_entries, particle_tree_entries, num_trees,
                ab_version, tcpt_version, vjj_version,
                file_path
            ) VALUES (
                :data_type, :campaign, :dataset_id, :e_tag,
                :num_executed_files, :num_input_events,
                :reco_tree_entries, :particle_tree_entries, :num_trees,
                :ab_version, :tcpt_version, :vjj_version,
                :file_path
            )
        """
        insert_values = {
            "data_type": metadata.data_type,
            "campaign": metadata.campaign,
            "dataset_id": metadata.dataset_id,
            "e_tag": metadata.e_tag,
            "num_executed_files": metadata.num_executed_files,
            "num_input_events": metadata.num_input_events,
            "reco_tree_entries": metadata.reco_tree_entries,
            "particle_tree_entries": metadata.particle_tree_entries,
            "num_trees": metadata.num_trees,
            "ab_version": metadata.ab_version,
            "tcpt_version": metadata.tcpt_version,
            "vjj_version": metadata.vjj_version,
            "file_path": metadata.file_path,
        }
        cur.execute(insert_query, insert_values)
        file_id = cur.lastrowid

        # Insert cutbookkeeper and update summary
        for (dsid, run_number, syst), weight_info in metadata.cutbookkeeper.items():
            cur.execute(
                """
                INSERT INTO cutbookkeeper (
                    file_id, dataset_id, run_number, syst,
                    total_events, sum_weights, sum_weights_sq
                ) VALUES (
                    :file_id, :dataset_id, :run_number, :syst,
                    :total_events, :sum_weights, :sum_weights_sq
                )
            """,
                {
                    "file_id": file_id,
                    "dataset_id": dsid,
                    "run_number": run_number,
                    "syst": syst,
                    "total_events": weight_info[0],
                    "sum_weights": weight_info[1],
                    "sum_weights_sq": weight_info[2],
                },
            )

            # Incrementally update campaign_summary
            cur.execute(
                """
                INSERT INTO campaign_summary (
                    campaign, dataset_id, syst,
                    num_executed_files, num_input_events,
                    total_events, sum_weights, sum_weights_sq
                ) VALUES (
                    :campaign, :dataset_id, :syst,
                    :num_executed_files, :num_input_events,
                    :total_events, :sum_weights, :sum_weights_sq
                )
                ON CONFLICT(campaign, dataset_id, syst) DO UPDATE SET
                    num_executed_files = num_executed_files + excluded.num_executed_files,
                    num_input_events = num_input_events + excluded.num_input_events,
                    total_events = total_events + excluded.total_events,
                    sum_weights = sum_weights + excluded.sum_weights,
                    sum_weights_sq = sum_weights_sq + excluded.sum_weights_sq
            """,
                {
                    "campaign": metadata.campaign,
                    "dataset_id": dsid,
                    "syst": syst,
                    "num_executed_files": metadata.num_executed_files,
                    "num_input_events": metadata.num_input_events,
                    "total_events": weight_info[0],
                    "sum_weights": weight_info[1],
                    "sum_weights_sq": weight_info[2],
                },
            )

        if commit:
            self.conn.commit()

        return file_id

    def remove_file(self, file_path: str, commit: bool = True) -> bool:
        cur = self.conn.cursor()

        # Get file id and campaign
        cur.execute(
            "SELECT id, campaign FROM file_metadata WHERE file_path = ?", (file_path,)
        )
        row = cur.fetchone()
        if not row:
            return True

        if self.exist_ok:
            return False

        file_id, campaign = row

        # Get cutbookkeeper entries
        cur.execute(
            """
            SELECT dataset_id, syst, total_events, sum_weights, sum_weights_sq
            FROM cutbookkeeper
            WHERE file_id = ?
        """,
            (file_id,),
        )
        cut_entries = cur.fetchall()

        # Decrement campaign_summary
        for dsid, syst, total_events, sum_weights, sum_weights_sq in cut_entries:
            cur.execute(
                """
                UPDATE campaign_summary
                SET total_events = total_events - ?,
                    sum_weights = sum_weights - ?,
                    sum_weights_sq = sum_weights_sq - ?
                WHERE campaign = ? AND dataset_id = ? AND syst = ?
            """,
                (total_events, sum_weights, sum_weights_sq, campaign, dsid, syst),
            )

        # Delete cutbookkeeper entries
        cur.execute("DELETE FROM cutbookkeeper WHERE file_id = ?", (file_id,))

        # Delete file metadata
        cur.execute("DELETE FROM file_metadata WHERE id = ?", (file_id,))

        if commit:
            self.conn.commit()

        return True

    def query_file_metadata(
        self,
        filters: Optional[dict[str, Union[Any, list[Any]]]] = None,
        columns: Optional[list[str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[tuple[Any, ...]]:
        cur = self.conn.cursor()
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM file_metadata"
        values: list[Any] = []

        if filters:
            clauses = []
            for col, val in filters.items():
                if isinstance(val, list) and val:  # handle IN queries
                    placeholders = ",".join("?" for _ in val)
                    clauses.append(f"{col} IN ({placeholders})")
                    values.extend(val)
                elif val is None:
                    clauses.append(f"{col} IS NULL")
                else:
                    clauses.append(f"{col} = ?")
                    values.append(val)
            query += " WHERE " + " AND ".join(clauses)

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += " LIMIT ?"
            values.append(limit)

        cur.execute(query, values)
        return cur.fetchall()

    def query_files(
        self,
        campaign: str | list[str] | None = None,
        dataset_ids: int | list[int] | None = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[str]:
        filters: dict[str, Any] = {}
        if campaign:
            filters["campaign"] = campaign
        if dataset_ids:
            filters["dataset_id"] = dataset_ids

        rows = self.query_file_metadata(
            columns=["file_path"],
            filters=filters,
            order_by=order_by,
            limit=limit,
        )
        return [row[0] for row in rows]


def generate_metadata_db(
    ntuple_files: list[str], output: str = "metadata.sqlite3", exist_ok: bool = False
) -> None:
    with FileSQLiteDB(output) as db:
        if exist_ok:
            # handle duplicates by caching all existing file path.
            db.handle_duplicates = False  # turn off internal check for duplication.
            cur = db.conn.cursor()
            cur.execute("SELECT file_path FROM file_metadata")
            existing_files = {row[0] for row in cur.fetchall()}
        else:
            existing_files = set()

        db.conn.execute("BEGIN")
        for f in ntuple_files:
            f = str(Path(f).resolve())
            if f in existing_files:
                continue
            db.insert_file(FileMetaData(f), commit=False)
        db.conn.commit()
