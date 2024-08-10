import boto3
import os

def backup_folder_to_s3(srcPath, bucket, dstPrefix, swipeLocal=True):
     
    s3_client = boto3.client("s3")
    
    for dir, _, files in os.walk(srcPath):
        sub_dir=dir.replace(srcPath, "")
        sub_dir=sub_dir[1:] if sub_dir.startswith("/") else sub_dir
        filePrefix = os.path.join(dstPrefix, sub_dir)

        for file in files:
            dir_file = os.path.join(dir, file)
            s3_client.upload_file(Filename=dir_file,
                    Bucket=bucket,
                    Key=os.path.join(filePrefix ,file).replace(" ", "_"))

            #if swipeLocal:
             #   os.remove(dir_file)

    print("Finished backig up directory to s3: {}".format(srcPath))
