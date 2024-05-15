import os
import re


def replace_type_hints(file_path):
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Decode the file data with error handling
    file_data = file_data.decode("utf-8", errors="ignore")

    # Regular expression pattern to find 'Dict[Type1, Type2] | None' and replace with 'Optional[Dict[Type1, Type2]]'.
    file_data = re.sub(
        r"Dict\[(\w+), (\w+)\]\s*\|\s*None", r"Optional[Dict[\1, \2]]", file_data
    )

    with open(file_path, "w") as file:
        file.write(file_data)


# Directory path
dir_path = "/Users/jakit/customers/aurelio/semantic-router"

# Traverse the directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith(".py"):
            replace_type_hints(os.path.join(root, file))
