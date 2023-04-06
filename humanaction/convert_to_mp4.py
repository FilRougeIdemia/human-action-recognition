import os

# specify the directory containing the .MOV files
directory = '/path/to/directory'

# iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.MOV'):
        # construct the full file path
        filepath = os.path.join(directory, filename)

        # construct the new file path with .mp4 extension
        new_filepath = os.path.splitext(filepath)[0] + '.mp4'

        # use ffmpeg to convert the file
        os.system(f'ffmpeg -i {filepath} {new_filepath}')