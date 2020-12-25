import boto3

def lambda_handler(event, context):
    
    client = boto3.client(service_name='glue')
    
    client.start_job_run(
        JobName = 'imba-glue'
        )
    
    return {
        'statusCode': 200,
    }
