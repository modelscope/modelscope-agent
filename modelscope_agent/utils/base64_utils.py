import base64
import os


def encode_file_to_base64(file_path: str):
    # Read the file content and encode it to base64
    with open(file_path, 'rb') as file_to_encode:
        encoded_content = base64.b64encode(file_to_encode.read())
        return encoded_content.decode('utf-8')


def decode_base64_to_file(base64_string: str, file_path: str):
    # Decode the base64 string to get the binary content
    file_content = base64.b64decode(base64_string)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the binary content to the file
    with open(file_path, 'wb') as output_file:
        output_file.write(file_content)


def encode_files_to_base64(file_paths: list):
    # Encode multiple files to base64 in a dict
    encoded_files = {}
    for file_path in file_paths:
        base64_image = encode_file_to_base64(file_path)
        encoded_files[os.path.basename(file_path)] = base64_image
    return encoded_files


def decode_base64_to_files(base64_files_dict: dict, output_dir: str):
    # Decode multiple base64 strings to files from a dict
    decoded_files = {}
    for file_name in base64_files_dict:
        file_path = os.path.join(output_dir, file_name)
        decode_base64_to_file(base64_files_dict[file_name], file_path)
        decoded_files[file_name] = file_path
    return decoded_files
