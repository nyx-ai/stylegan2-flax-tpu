from urllib.request import Request, urlopen
import urllib.error
try:
    from google.cloud import storage
except ImportError:
    print('Install `pip install google-cloud-storage` to get full support for GCS')


def get_vm_name():
    gcp_metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance-id"
    req = Request(gcp_metadata_url)
    req.add_header('Metadata-Flavor', 'Google')
    instance_id = None
    try:
        with urlopen(req) as url:
            instance_id = url.read().decode()
    except urllib.error.URLError:
        # metadata.google.internal not reachable
        pass
    return instance_id


def get_current_region():
    gcp_metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance-location"
    req = Request(gcp_metadata_url)
    req.add_header('Metadata-Flavor', 'Google')
    current_region = None
    try:
        with urlopen(req) as url:
            current_region = url.read().decode()
    except urllib.error.URLError:
        # metadata.google.internal not reachable: use dev
        pass
    return current_region


def get_bucket_location(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    loc = bucket.location
    return loc.lower()


def validate_dir_is_in_current_region(save_dir):
    if not save_dir.startswith('gs://'):
        return True
    bucket_name = save_dir.split('gs://')[1].split('/')[0]
    bucket_location = get_bucket_location(bucket_name)
    current_region = get_current_region()
    current_region = current_region[:-2]  # removes availability zone
    if bucket_location != current_region:
        raise Exception(f'Bucket location is {bucket_location} but running currently in {current_region}!')
    return True
