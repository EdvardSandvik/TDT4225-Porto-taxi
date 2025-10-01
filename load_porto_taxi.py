"""
Load Porto taxi CSV into MySQL with helpful debug prints.

This script:
- Reads CSV in chunks to limit memory use.
- Parses POLYLINE safely with ast.literal_eval and stores as JSON.
- Skips duplicate primary keys using INSERT IGNORE and logs skipped rows.
- Prints progress and detailed errors to help debugging when things go wrong.
"""
import os
import ast
import json
import traceback
import pandas as pd
import mysql.connector
from DbConnector import DbConnector

def load_csv_to_mysql(csv_path: str, chunksize: int = 1000):
    """
    Load a CSV file into the `trips` MySQL table.

    Args:
        csv_path: path to the CSV file or to a directory containing 'porto.csv'.
        chunksize: number of rows to read per pandas chunk.
    """
    print(f"[INFO] Starting load_csv_to_mysql with csv_path={csv_path!r}, chunksize={chunksize}")
    csv_path = os.path.expanduser(csv_path)

    # If user passed a directory, assume the CSV filename is porto.csv
    if os.path.isdir(csv_path):
        csv_path = os.path.join(csv_path, "porto.csv")
        print(f"[INFO] Given path is a directory, using file: {csv_path}")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Create DB connector and obtain a cursor
    conn = DbConnector()
    print(f"[INFO] Created DbConnector: {conn!r}")
    try:
        cursor = conn.db_connection.cursor()
        print("[INFO] Obtained cursor via conn.db_connection.cursor()")
    except Exception:
        try:
            cursor = conn.cursor()
            print("[INFO] Obtained cursor via conn.cursor()")
        except Exception:
            print("[ERROR] Failed to obtain a cursor from DbConnector. Conn object methods:", dir(conn))
            raise

    # Use INSERT IGNORE to skip duplicate primary keys (TRIP_ID) instead of raising IntegrityError.
    insert_query = """
        INSERT IGNORE INTO trips (TRIP_ID, CALL_TYPE, ORIGIN_CALL, ORIGIN_STAND,
                           TAXI_ID, TIMESTAMP, DAYTYPE, MISSING_DATA, POLYLINE)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    total_attempted = 0
    total_success = 0
    total_skipped_duplicates = 0
    chunk_index = 0

    # Read CSV in chunks to avoid memory spikes
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk_index += 1
        rows_in_chunk = len(chunk)
        print(f"[INFO] Processing chunk {chunk_index} with {rows_in_chunk} rows")

        success_in_chunk = 0
        attempt_in_chunk = 0
        skipped_in_chunk = 0

        for _, row in chunk.iterrows():
            attempt_in_chunk += 1
            total_attempted += 1

            # Safely parse POLYLINE to JSON; handle NaN/empty
            poly = row.get("POLYLINE", None)
            poly_json = None
            try:
                if pd.notna(poly):
                    parsed = ast.literal_eval(poly) if isinstance(poly, str) else poly
                    poly_json = json.dumps(parsed)
            except Exception:
                print(f"[WARN] Failed to parse POLYLINE for TRIP_ID={row.get('TRIP_ID')!r}. Storing raw value.")
                try:
                    poly_json = json.dumps(poly)
                except Exception:
                    poly_json = None

            params = (
                row.get("TRIP_ID"),
                row.get("CALL_TYPE"),
                row.get("ORIGIN_CALL"),
                row.get("ORIGIN_STAND"),
                row.get("TAXI_ID"),
                row.get("TIMESTAMP"),
                row.get("DAYTYPE"),
                row.get("MISSING_DATA"),
                poly_json
            )

            # Execute and handle duplicate-key cases gracefully
            try:
                cursor.execute(insert_query, params)
                # For mysql-connector, rowcount == 1 if a row was inserted, 0 if ignored (duplicate)
                if getattr(cursor, "rowcount", None) == 1:
                    success_in_chunk += 1
                    total_success += 1
                else:
                    skipped_in_chunk += 1
                    total_skipped_duplicates += 1
                    print(f"[INFO] Skipped duplicate TRIP_ID={row.get('TRIP_ID')!r} (chunk {chunk_index}, row {attempt_in_chunk})")
            except mysql.connector.errors.IntegrityError as ie:
                # Should be rare because INSERT IGNORE should prevent duplicates, but log if it occurs
                print(f"[WARN] IntegrityError on TRIP_ID={row.get('TRIP_ID')!r}: {ie!r}")
                traceback.print_exc()
                skipped_in_chunk += 1
                total_skipped_duplicates += 1
                continue
            except Exception as e:
                print(f"[ERROR] DB execute failed on chunk {chunk_index}, row {attempt_in_chunk}. TRIP_ID={row.get('TRIP_ID')!r}")
                print(f"[ERROR] Exception: {e!r}")
                traceback.print_exc()
                # Re-raise to let caller see serious unexpected errors
                raise

        # Commit after finishing the chunk and log commit status
        try:
            if hasattr(conn, "db_connection") and hasattr(conn.db_connection, "commit"):
                conn.db_connection.commit()
                print(f"[INFO] Committed chunk {chunk_index} via conn.db_connection.commit()")
            elif hasattr(conn, "commit"):
                conn.commit()
                print(f"[INFO] Committed chunk {chunk_index} via conn.commit()")
            else:
                print(f"[WARN] No commit method found on connection after chunk {chunk_index}")
        except Exception as e:
            print(f"[ERROR] Commit failed after chunk {chunk_index}: {e!r}")
            traceback.print_exc()
            raise

        print(f"[INFO] Chunk {chunk_index} summary: attempted={attempt_in_chunk}, inserted={success_in_chunk}, skipped_duplicates={skipped_in_chunk}")

    # Close connection gracefully and log
    try:
        if hasattr(conn, "close_connection"):
            conn.close_connection()
            print("[INFO] Closed connection via conn.close_connection()")
        elif hasattr(conn, "close"):
            conn.close()
            print("[INFO] Closed connection via conn.close()")
        elif hasattr(conn, "db_connection") and hasattr(conn.db_connection, "close"):
            conn.db_connection.close()
            print("[INFO] Closed connection via conn.db_connection.close()")
        else:
            print("[WARN] No close method found on connection object")
    except Exception as e:
        print(f"[WARN] Exception while closing connection: {e!r}")
        traceback.print_exc()

    print(f"[INFO] Done. Total attempted: {total_attempted}, total inserted: {total_success}, total skipped_duplicates: {total_skipped_duplicates}")

if __name__ == "__main__":
    # Default path to your local porto folder; script will look for porto.csv inside it
    default_folder = "/Users/edvardsandvik/Documents/NTNU/2025:26/TDT4225 Store distribuerte datamengder/porto"
    load_csv_to_mysql(default_folder)