import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import ast
import os

load_dotenv()

filename = f'../logs/{pd.Timestamp.now().strftime("%Y-%m-%d")}.log'
os.makedirs(os.path.dirname(filename), exist_ok=True)

churro = os.getenv("CHURRO")
engine = create_engine(churro)

def filter_confidence_data(df):

    df = df[df['confidence'] > 0.6] #Filter out detections with confidence <= 0.6 (adjust threshold as needed)

    with open(filename, 'a') as log_file:
        log_file.write(f"--Filtered Detections--\n")
        log_file.write(f"Filtered detections at {pd.Timestamp.now()}\n")
        log_file.write(f"Filtered {len(df)} detections with confidence > 0.6\n")

    return df

def insert_data(df, table_name):

    try:
        df.to_sql(table_name, con=engine, if_exists='append', index=False) #Insert data into the specified table, appending to existing data
        print(f"Data inserted successfully into {table_name} table.")

        with open(filename, 'a') as log_file:
            log_file.write(f"--Data Insertion--\n")
            log_file.write(f"Inserted {len(df)} records into {table_name} table at {pd.Timestamp.now()}\n")

    except Exception as e:
        print(f"Error inserting data into {table_name} table: {e}")

        with open(filename, 'a') as log_file:
            log_file.write(f"--Data Insertion Error--\n")
            log_file.write(f"Error inserting data into {table_name} table at {pd.Timestamp.now()}: {e}\n")


def build_traffic_dataframe_simple(detections, zones, time_freq="1h"):
   
    #Auxiliar functions that parse the coord_lims column in zones and detections, and extract store_id and camera_prefix from camera_id in detections
    def parse_rect(text):

        if pd.isna(text):
            return None, None, None, None

        d = ast.literal_eval(text)
        return d["x_min"], d["x_max"], d["y_min"], d["y_max"]

    def parse_point(text):

        if pd.isna(text):
            return None, None
        
        x, y = ast.literal_eval(text)
        return float(x), float(y)

    def get_store_id(camera_id):

        return camera_id.split("_")[-1]

    def get_camera_prefix(camera_id):

        return camera_id.split("_")[0].replace("CM", "")


    zones[["x_min", "x_max", "y_min", "y_max"]] = zones["coord_lims"].apply(
        lambda x: pd.Series(parse_rect(x))
    )

    zones["zone_prefix"] = zones["zone_id"].str.split("_").str[0]

    #We filter detections to only include those classified as "Client", and we extract timestamp, store_id, camera_prefix, and coordinates (x, y) from the coord_lims column
    detections = detections[detections["class_object"] == "Client"].copy()
    detections["timestamp"] = pd.to_datetime(detections["timestamp"])
    detections["store_id"] = detections["camera_id"].apply(get_store_id)
    detections["camera_prefix"] = detections["camera_id"].apply(get_camera_prefix)
    detections[["x", "y"]] = detections["coord_lims"].apply(
        lambda x: pd.Series(parse_point(x))
    )

    #Function to assign zone_id to each detection
    def assign_zone(row):
        store_zones = zones[zones["store_id"] == row["store_id"]]

        #First trying to match by coordinates
        if pd.notna(row["x"]) and pd.notna(row["y"]):
            match = store_zones[
                (store_zones["x_min"] <= row["x"]) &
                (store_zones["x_max"] >= row["x"]) &
                (store_zones["y_min"] <= row["y"]) &
                (store_zones["y_max"] >= row["y"])
            ]
            if not match.empty:
                return match.iloc[0]["zone_id"]

        #If that fails, trying to match by camera prefix
        match = store_zones[store_zones["zone_prefix"] == row["camera_prefix"]]

        if not match.empty:
            return match.iloc[0]["zone_id"]

        #If both fail, return None
        return None

    #Assign zone_id to each detection using the assign_zone function
    detections["zone_id"] = detections.apply(assign_zone, axis=1)
    detections = detections.dropna(subset=["zone_id"]).copy()

    #We create date_time and visit_day columns by flooring the timestamp to the specified time frequency and extracting the date
    detections["date_time"] = detections["timestamp"].dt.floor(time_freq)
    detections["visit_day"] = detections["timestamp"].dt.date

    #We group detections by visit_day, store_id, and tracking_id to identify individual visits
    visits = (
        detections.groupby(["visit_day", "store_id", "tracking_id"])
        .agg(
            visit_start=("timestamp", "min"),
            visit_end=("timestamp", "max"),
            distinct_zones=("zone_id", "nunique")
        )
        .reset_index()
    )

    #We calculate the average time in store for each visit and whether it is a bounce (only one distinct zone visited)
    visits["average_time_in_store"] = (
        (visits["visit_end"] - visits["visit_start"]).dt.total_seconds() / 60
    )

    visits["is_bounce"] = (visits["distinct_zones"] == 1).astype(int)

    #We merge the visits information back to the detections to assign average_time_in_store and is_bounce to each detection, and we drop duplicates to have one record per visit per zone
    visit_zone_time = (
        detections[["date_time", "visit_day", "store_id", "zone_id", "tracking_id"]]
        .drop_duplicates()
        .merge(
            visits[["visit_day", "store_id", "tracking_id", "average_time_in_store", "is_bounce"]],
            on=["visit_day", "store_id", "tracking_id"],
            how="left"
        )
    )

    #We start building the traffic dataframe
    traffic = (
        visit_zone_time.groupby(["date_time", "store_id", "zone_id"])
        .agg(
            visitor_count=("tracking_id", "nunique"),
            average_time_in_store=("average_time_in_store", "mean"),
            bounce_rate=("is_bounce", "mean")
        )
        .reset_index()
    )

    #We calculate the peak_people metric as the maximum number of unique tracking_ids in the same date_time, store_id, and zone_id
    peak = (
        detections.groupby(["date_time", "store_id", "zone_id"])
        .agg(peak_people=("tracking_id", "nunique"))
        .reset_index()
    )

    #We merge the peak information into the traffic dataframe
    traffic = traffic.merge(
        peak,
        on=["date_time", "store_id", "zone_id"],
        how="left"
    )

    #We convert bounce_rate to percentage
    traffic["bounce_rate"] = traffic["bounce_rate"] * 100

    #traffic_id
    traffic["traffic_id"] = (
        "TRF_"
        + traffic["store_id"].astype(str)
        + "_"
        + traffic["zone_id"].astype(str)
        + "_"
        + traffic["date_time"].dt.strftime("%Y%m%d%H%M%S")
    )

    traffic = traffic[
        [
            "traffic_id",
            "date_time",
            "store_id",
            "zone_id",
            "visitor_count",
            "average_time_in_store",
            "peak_people",
            "bounce_rate"
        ]
    ].sort_values(["date_time", "store_id", "zone_id"]).reset_index(drop=True)

    with open(filename, 'a') as log_file:
        log_file.write(f"--Traffic DataFrame Built--\n")
        log_file.write(f"Built traffic dataframe with {len(traffic)} records at {pd.Timestamp.now()}\n")

    return traffic

def build_joined_datasets(df_detections, df_traffic, df_stores, df_zones, df_cameras, df_sales, df_soldproducts):
    # infrastructure dataset (camera + zones + stores)
    infrastructure = (
        df_cameras.merge(df_zones, on='zone_id', how='left')
               .merge(df_stores, on='store_id', how='left')
               .rename(columns={
                   'lims': 'camera_lims',
                   'coord_lims': 'zone_lims',
                   'condition': 'camera_condition'  # to avoid confusions
               })
    )
    # reorder
    infrastructure = infrastructure[[
        'store_id', 'store_name', 'city',
        'zone_id', 'zone_name', 'zone_type', 'zone_lims',
        'camera_id', 'model','camera_condition','camera_lims', 
        'installation_date', 'm2', 'max_capacity'
    ]]

    # revenue dataset (soldproducts + sales + stores)
    revenue = (
        df_soldproducts.merge(df_sales, on='ticket_id', how='inner')
                    .merge(df_stores, on='store_id', how='left')
                    .rename(columns={
                        'name': 'product_name',
                        'category': 'product_category',
                        'price': 'product_price'
                    })
    )
    # reorder
    revenue = revenue[[
        'ticket_id', 'timestamp',
        'store_id','store_name', 'city',
        'product_id', 'product_name', 'product_category', 'product_price',
        'total_euros', 'product_amount',
        'checkout_number', 'zone_id',
        'm2', 'max_capacity'
    ]]

    # validation dataset
    df_detections['timestamp'] = pd.to_datetime(df_detections['timestamp']) # datetime format for merging
    df_traffic['date_time'] = pd.to_datetime(df_traffic['date_time'])

    # create the hour_key for the merge
    df_detections['hour_key'] = df_detections['timestamp'].dt.floor('h')
    df_traffic['hour_key'] = df_traffic['date_time'].dt.floor('h')

    # detections + cameras
    detections_with_zones = pd.merge(
        df_detections, 
        df_cameras[['camera_id', 'zone_id']], 
        on='camera_id', 
        how='left'
    )

    # now that we have the zone_id, we add traffic
    validation = pd.merge(
        detections_with_zones, 
        df_traffic, 
        on=['zone_id', 'hour_key'], 
        how='inner'
    )

    # reorder
    validation = validation[[
        'detection_id','tracking_id',
        'hour_key', 'timestamp', 'date_time', 'traffic_id',
        'store_id', 'zone_id', 'camera_id',
        'class_object',
        'visitor_count', 'peak_people',
        'confidence', 'coord_lims',
        'average_time_in_store', 'bounce_rate'
    ]]

    # add IDs
    infrastructure["infrastructure_id"] = range(1, len(infrastructure) + 1)
    revenue["revenue_id"] = range(1, len(revenue) + 1)
    validation["validation_id"] = range(1, len(validation) + 1)
    with open(filename, 'a') as log_file:
        log_file.write(f"--Joined Datasets--\n")
        log_file.write(f"3 datasets have been created with joins at {pd.Timestamp.now()}\n")

    return infrastructure, revenue, validation

def main():

    if not engine:
        print("Error: No se pudo conectar a la base de datos.")

        with open(filename, 'a') as log_file:
            log_file.write(f"--Database Connection Error--\n")
            log_file.write(f"Failed to connect to the database at {pd.Timestamp.now()}\n")

        return
    
    df_stores = pd.read_csv('./data/stores.csv')
    df_sales = pd.read_csv('./data/sales.csv')
    df_zones = pd.read_csv('./data/zones.csv')
    df_cameras = pd.read_csv('./data/cameras.csv')
    df_detections = pd.read_csv('./data/detections.csv')
    df_soldproducts = pd.read_csv('./data/soldproducts.csv')

    insert_data(df_stores, "stores")
    insert_data(df_zones, "zones")
    insert_data(df_cameras, "cameras")
    insert_data(df_sales, "sales")
    insert_data(df_soldproducts, "soldproducts")

    df_detections = filter_confidence_data(df_detections)
    df_traffic = build_traffic_dataframe_simple(df_detections, df_zones)


    insert_data(df_detections, "detections")
    insert_data(df_traffic, "traffic")
    
    infrastructure, revenue, validation = build_joined_datasets(df_detections, df_traffic, df_stores, df_zones, df_cameras, df_sales, df_soldproducts)
    insert_data(infrastructure, "infrastructure")
    insert_data(revenue, "revenue")
    insert_data(validation, "validation")

if __name__ == "__main__":
    main()