

# for table prd_features
import json
import boto3
import time

athena_client = boto3.client('athena')

def lambda_handler(event, context):
    # TODO implement
    database = event['database']
    query_output = event['query_output']
    
    query1 = """
    DROP TABLE IF EXISTS prd_features
    """
    
    query2 = """
    CREATE TABLE prd_features WITH 
    (external_location = 's3://imba4sophie/features/prd_features/',
    format='parquet')
    as(
    SELECT product_id,
    Count(*) AS prod_orders,
    Sum(reordered) AS prod_reorders,
    Sum(CASE WHEN product_seq_time = 1 THEN 1 ELSE 0 END) AS prod_first_orders, Sum(CASE WHEN product_seq_time = 2 THEN 1 ELSE 0 END) AS prod_second_orders
    FROM (
    SELECT *, 
    Rank() OVER (partition BY user_id, product_id ORDER BY user_id, order_number) AS product_seq_time
    FROM order_products_prior) 
    GROUP BY product_id
    )
    """
    response1 = athena_client.start_query_execution(
        QueryString = query1,
        QueryExecutionContext={'Database':database},
        ResultConfiguration = {'OutputLocation': query_output}
        )
        
    #sleep 10s to make sure the table is successfully dropped
    time.sleep (10)
    
    response2 = athena_client.start_query_execution(
        QueryString = query2,
        QueryExecutionContext = {'Database': database},
        ResultConfiguration = {'OutputLocation':query_output}
        )
    
    #get the query execution id
    execution_id = response2['QueryExecutionId']
    
    while True:
        stats = athena_client.get_query_execution(QueryExecutionId=execution_id) 
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.2) # 200ms
    return {
        'statusCode': status
    }



# for table up_features
import json
import boto3
import time

athena_client = boto3.client('athena')

def lambda_handler(event, context):
    # TODO implement
    database = event['database']
    query_output = event['query_output']
    
    query1 = """
    DROP TABLE IF EXISTS up_features
    """
    
    query2 = """
    CREATE TABLE up_features WITH 
    (external_location = 's3://imba4sophie/features/up_features/',
    format='parquet')
    as(
    SELECT user_id, product_id,
    Count(*) AS up_orders,
    Min(order_number) AS up_first_order, Max(order_number) AS up_last_order, Avg(add_to_cart_order) AS up_average_cart_position
    FROM order_products_prior 
    GROUP BY user_id,product_id
    )
    """
    response1 = athena_client.start_query_execution(
        QueryString = query1,
        QueryExecutionContext={'Database':database},
        ResultConfiguration = {'OutputLocation': query_output}
        )
        
    #sleep 10s to make sure the table is successfully dropped
    time.sleep (10)
    
    response2 = athena_client.start_query_execution(
        QueryString = query2,
        QueryExecutionContext = {'Database': database},
        ResultConfiguration = {'OutputLocation':query_output}
        )
    
    #get the query execution id
    execution_id = response2['QueryExecutionId']
    
    while True:
        stats = athena_client.get_query_execution(QueryExecutionId=execution_id) 
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.2) # 200ms
    return {
        'statusCode': status
    }

# for table user_features_2
import json
import boto3
import time

athena_client = boto3.client('athena')

def lambda_handler(event, context):
    # TODO implement
    database = event['database']
    query_output = event['query_output']
    
    query1 = """
    DROP TABLE IF EXISTS user_features_2
    """
    
    query2 = """
    CREATE TABLE user_features_2 WITH 
    (external_location = 's3://imba4sophie/features/user_features_2/',
    format='parquet')
    as(
    SELECT user_id,
    Count(*) AS user_total_products,
    Count(DISTINCT product_id) AS user_distinct_products,
    Sum(CASE WHEN reordered = 1 THEN 1 ELSE 0 END) / Cast(Sum(CASE WHEN order_number > 1 THEN 1 ELSE 0 END) AS DOUBLE) AS user_reorder_ratio
    FROM order_products_prior 
    GROUP BY user_id
    )
    """
    response1 = athena_client.start_query_execution(
        QueryString = query1,
        QueryExecutionContext={'Database':database},
        ResultConfiguration = {'OutputLocation': query_output}
        )
        
    #sleep 10s to make sure the table is successfully dropped
    time.sleep (10)
    
    response2 = athena_client.start_query_execution(
        QueryString = query2,
        QueryExecutionContext = {'Database': database},
        ResultConfiguration = {'OutputLocation':query_output}
        )
    
    #get the query execution id
    execution_id = response2['QueryExecutionId']
    
    while True:
        stats = athena_client.get_query_execution(QueryExecutionId=execution_id) 
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.2) # 200ms
    return {
        'statusCode': status
    }

# for table user_features_1
import json
import boto3
import time

athena_client = boto3.client('athena')

def lambda_handler(event, context):
    # TODO implement
    database = event['database']
    query_output = event['query_output']
    
    query1 = """
    DROP TABLE IF EXISTS user_features_1
    """
    
    query2 = """
    CREATE TABLE user_features_1 WITH 
    (external_location = 's3://imba4sophie/features/user_features_1/',
    format='parquet')
    as(
    SELECT user_id,
    Max(order_number) AS user_orders, 
    Sum(days_since_prior_order) AS user_period, 
    Avg(days_since_prior_order) AS user_mean_days_since_prior
    FROM order_products_prior
    GROUP BY user_id
    )
    """
    response1 = athena_client.start_query_execution(
        QueryString = query1,
        QueryExecutionContext={'Database':database},
        ResultConfiguration = {'OutputLocation': query_output}
        )
        
    #sleep 10s to make sure the table is successfully dropped
    time.sleep (10)
    
    response2 = athena_client.start_query_execution(
        QueryString = query2,
        QueryExecutionContext = {'Database': database},
        ResultConfiguration = {'OutputLocation':query_output}
        )
    
    #get the query execution id
    execution_id = response2['QueryExecutionId']
    
    while True:
        stats = athena_client.get_query_execution(QueryExecutionId=execution_id) 
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.2) # 200ms
    return {
        'statusCode': status
    }
    
    
# for table order_products_prior
import json
import boto3
import time

athena_client = boto3.client('athena')

def lambda_handler(event, context):
    # TODO implement
    database = event['database']
    query_output = event['query_output']
    
    query1 = """
    DROP TABLE IF EXISTS order_products_prior
    """
    
    query2 = """
    CREATE TABLE order_products_prior WITH 
    (external_location = 's3://imba4sophie/features/order_products_prior/',
    format='parquet')
    as(
    select a.*,
    b.product_id,
    b.add_to_cart_order,
    b.reordered FROM orders a
    JOIN order_products b
    ON a.order_id = b.order_id
    WHERE a.eval_set = 'prior'
    )
    """
    response1 = athena_client.start_query_execution(
        QueryString = query1,
        QueryExecutionContext={'Database':database},
            ResultConfiguration = {'OutputLocation': query_output}
        )
        
    #sleep 10s to make sure the table is successfully dropped
    time.sleep (10)
    
    response2 = athena_client.start_query_execution(
        QueryString = query2,
        QueryExecutionContext = {
            'Database': database
        },
        ResultConfiguration = {
            'OutputLocation':query_output
        }
        )
    
    #get the query execution id
    execution_id = response2['QueryExecutionId']
    
    while True:
        stats = athena_client.get_query_execution(QueryExecutionId=execution_id) 
        status = stats['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(0.2) # 200ms
    return {
        'statusCode': status
    }
    
# to remove feature file    
import json
import boto3

def lambda_handler(event, context):
    # TODO implement
    print(event)
    bucket = event['bucket']
    prefix = event['prefix']
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    
    for key in bucket.objects.filter(Prefix = prefix):
        key.delete()
    return {
        'statusCode': 200
    }
    
    
