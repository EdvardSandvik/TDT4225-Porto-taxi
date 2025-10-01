from DbConnector import DbConnector
from tabulate import tabulate
import pandas as pd
import numpy as np


class PortoTaxiEDA:
    def __init__(self):
        # connect to DB
        self.connection = DbConnector()
        self.db_connection = self.connection.db_connection
        self.cursor = self.connection.cursor

    def count_basic_stats(self):
        """Count trips, taxis, and GPS points"""
        queries = {
            "Number of taxis": "SELECT COUNT(DISTINCT TAXI_ID) FROM trips",
            "Number of trips": "SELECT COUNT(*) FROM trips",
            "Total GPS points": "SELECT SUM(JSON_LENGTH(POLYLINE)) FROM trips"
        }
        print("\n=== Basic Stats ===")
        for desc, query in queries.items():
            self.cursor.execute(query)
            result = self.cursor.fetchone()[0]
            print(f"{desc}: {result}")

    def trip_length_stats(self):
        """Trip duration and length statistics (based on GPS points)"""
        query = "SELECT JSON_LENGTH(POLYLINE) AS trip_points FROM trips"
        self.cursor.execute(query)
        rows = [r[0] for r in self.cursor.fetchall()]
        series = pd.Series(rows)

        stats = {
            "Min points": series.min(),
            "Max points": series.max(),
            "Avg points": series.mean(),
            "Median points": series.median()
        }

        print("\n=== Trip Length Stats (GPS points) ===")
        print(tabulate([stats.values()], headers=stats.keys(), floatfmt=".2f"))

    def invalid_trips(self):
        """Trips with fewer than 3 GPS points"""
        query = "SELECT COUNT(*) FROM trips WHERE JSON_LENGTH(POLYLINE) < 3"
        self.cursor.execute(query)
        count = self.cursor.fetchone()[0]
        print(f"\nInvalid trips (<3 GPS points): {count}")

    def stationary_trips(self):
        """Trips where start and end points are the same"""
        query = """
        SELECT COUNT(*) 
        FROM trips
        WHERE JSON_LENGTH(POLYLINE) >= 2
        AND JSON_EXTRACT(POLYLINE, '$[0]') = 
            JSON_EXTRACT(POLYLINE, CONCAT('$[', JSON_LENGTH(POLYLINE)-1, ']'));
        """
        self.cursor.execute(query)
        count = self.cursor.fetchone()[0]
        print(f"\nStationary trips (start=end): {count}")


    def call_type_distribution(self):
        """Distribution of call types (A, B, C)"""
        query = """
        SELECT CALL_TYPE, COUNT(*) AS num_trips
        FROM trips
        GROUP BY CALL_TYPE
        ORDER BY num_trips DESC;
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print("\n=== Distribution of Call Types ===")
        print(tabulate(rows, headers=self.cursor.column_names))

    def day_type_distribution(self):
        """Distribution of trips by day type (A,B,C)"""
        query = """
        SELECT DAYTYPE, COUNT(*) AS num_trips
        FROM trips
        GROUP BY DAYTYPE;
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print("\n=== Distribution of Day Types ===")
        print(tabulate(rows, headers=self.cursor.column_names))

    def missing_data_stats(self):
        """Check how many trips have missing GPS data"""
        query = """
        SELECT MISSING_DATA, COUNT(*) AS num_trips,
               AVG(JSON_LENGTH(POLYLINE)) AS avg_points
        FROM trips
        GROUP BY MISSING_DATA;
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print("\n=== Trips with Missing Data ===")
        print(tabulate(rows, headers=self.cursor.column_names, floatfmt=".2f"))

    def trips_per_taxi(self):
        """Distribution of trips per taxi + top 20 taxis"""
        query = """
        SELECT TAXI_ID, COUNT(*) AS num_trips
        FROM trips
        GROUP BY TAXI_ID
        ORDER BY num_trips DESC
        LIMIT 20;
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print("\n=== Top 20 Taxis by Number of Trips ===")
        print(tabulate(rows, headers=self.cursor.column_names))

        query_hist = """
        SELECT COUNT_TRIPS, COUNT(*) AS num_taxis
        FROM (
            SELECT TAXI_ID, COUNT(*) AS COUNT_TRIPS
            FROM trips
            GROUP BY TAXI_ID
        ) sub
        GROUP BY COUNT_TRIPS
        ORDER BY COUNT_TRIPS;
        """
        self.cursor.execute(query_hist)
        rows = self.cursor.fetchall()
        print("\n=== Histogram of Trips per Taxi ===")
        print(tabulate(rows, headers=self.cursor.column_names))

    def close(self):
        self.connection.close_connection()

    
    def avg_trip_length_top_taxis(self, limit=20):
        """Top taxis by number of trips with their average trip length"""
        query = f"""
        SELECT TAXI_ID,
            COUNT(*) AS num_trips,
            AVG(JSON_LENGTH(POLYLINE)) AS avg_points,
            MIN(JSON_LENGTH(POLYLINE)) AS min_points,
            MAX(JSON_LENGTH(POLYLINE)) AS max_points
        FROM trips
        GROUP BY TAXI_ID
        ORDER BY num_trips DESC
        LIMIT {limit};
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print(f"\n=== Avg Trip Length for Top {limit} Taxis ===")
        print(tabulate(rows, headers=self.cursor.column_names, floatfmt=".2f"))

    
    def avg_trip_length_by_call_type(self):
        """Average trip length (GPS points ~ duration) per call type"""
        query = """
        SELECT CALL_TYPE,
            COUNT(*) AS num_trips,
            AVG(JSON_LENGTH(POLYLINE)) AS avg_points,
            MIN(JSON_LENGTH(POLYLINE)) AS min_points,
            MAX(JSON_LENGTH(POLYLINE)) AS max_points
        FROM trips
        GROUP BY CALL_TYPE
        ORDER BY num_trips DESC;
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print("\n=== Avg Trip Length by Call Type ===")
        print(tabulate(rows, headers=self.cursor.column_names, floatfmt=".2f"))
    
    def trips_by_hour(self):
        """Distribution of trips across hours of the day"""
        query = """
        SELECT 
            HOUR(FROM_UNIXTIME(TIMESTAMP)) AS hour_of_day,
            COUNT(*) AS num_trips
        FROM trips
        GROUP BY hour_of_day
        ORDER BY hour_of_day;
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        print("\n=== Trips by Hour of Day ===")
        print(tabulate(rows, headers=self.cursor.column_names))




def main():
    program = None
    try:
        program = PortoTaxiEDA()
        #program.count_basic_stats()
        #program.trip_length_stats()
        #program.invalid_trips()
        #program.stationary_trips()
        #program.call_type_distribution()
        #program.day_type_distribution()
        #program.missing_data_stats()
        #program.trips_per_taxi()
        #program.avg_trip_length_top_taxis(limit=20)
        #program.avg_trip_length_by_call_type()
        program.trips_by_hour()
    except Exception as e:
        print("ERROR:", e)
    finally:
        if program:
            program.close()


if __name__ == '__main__':
    main()
