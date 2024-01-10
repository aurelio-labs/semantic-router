import re
import os


def replace_type_hints(file_path):
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Decode the file data with error handling
    file_data = file_data.decode("utf-8", errors="ignore")

    # Regular expression pattern to find '| None' and replace with 'Optional'
    file_data = re.sub(r"(\w+)\s*\|\s*None", r"Optional[\1]", file_data)

    with open(file_path, "w") as file:
        file.write(file_data)


# Walk through the repository and update all .py files
for root, dirs, files in os.walk("/Users/jakit/customers/aurelio/semantic-router"):
    for file in files:
        if file.endswith(".py"):
            replace_type_hints(os.path.join(root, file))
