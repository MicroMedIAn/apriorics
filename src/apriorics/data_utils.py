import sqlite3

from ordered_set import OrderedSet
from pathaia.util.paths import get_files


def reverse_mapping(type_id, mapping):
    slide_type = "HE" if type_id < 13 else "IHC"
    for ihc_type, id in mapping[slide_type].items():
        if id == type_id:
            return slide_type, ihc_type
    raise ValueError


def get_info_from_filename(filename, mapping):
    split_name = filename.split("-")
    block = split_name[0]
    id = int(split_name[-1].split("_")[0])
    slide_type, ihc_type = reverse_mapping(id, mapping)

    return {"block": block, "slide_type": slide_type, "ihc_type": ihc_type}


def create_database(database, table):
    with sqlite3.connect(database) as con:
        cmd = f"""
            CREATE TABLE {table}(
                id TEXT NOT NULL PRIMARY KEY,
                block TEXT NOT NULL,
                slide_type TEXT NOT NULL,
                ihc_type TEXT NOT NULL,
                path TEXT NOT NULL)
        """
        try:
            con.execute(cmd)
        except sqlite3.OperationalError:
            return
        con.execute(
            f"CREATE INDEX slide_index ON {table} (block, slide_type, ihc_type)"
        )


def update_database(
    database, registry, table, data_path, recurse=False, extension=".svs"
):
    files = get_files(data_path, extensions=extension, recurse=recurse)
    file_entries = files.map(lambda x: (*x.parts[-3:-1], x.stem))

    with sqlite3.connect(registry) as con:
        file_ids = OrderedSet()
        cmd = """
        SELECT id FROM slides
        WHERE ihc_type = ?
        AND slide_type = ?
        AND block = ?
        """
        for entry in file_entries:
            file_ids.add(con.execute(cmd, entry).fetchone()[0])

    with sqlite3.connect(database) as con:
        db_ids = OrderedSet(
            map(lambda x: x[0], con.execute(f"SELECT id FROM {table}").fetchall())
        )

        to_add = file_ids - db_ids
        to_remove = db_ids - file_ids
        con.execute(
            f"DELETE FROM {table} WHERE id IN ({','.join(['?']*len(to_remove))})",
            to_remove,
        )

        for id in sorted(to_add):
            idx = file_ids.index(id)
            entries = {
                k: v
                for k, v in zip(("ihc_type", "slide_type", "block"), file_entries[idx])
            }
            entries["id"] = id
            entries["path"] = str(files[idx].relative_to(data_path))
            cmd = f"""
            INSERT INTO {table} (id, block, slide_type, ihc_type, path)
            VALUES (:id, :block, :slide_type, :ihc_type, :path)
            """
            con.execute(cmd, entries)
