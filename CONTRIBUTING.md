# Contributing to the Semantic Router

The Aurelio Team welcome and encourage any contributions to the Semantic Router, big or small. Please feel free to contribute to new features, bug fixes or documenation. We're always eager to hear your suggestions.

Please follow these guidelines when making a contribution:
1. Before making any changes, [check here for related issues](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md).
2. If no related issue exists yet, please create one and suggest your changes. Checking in with the team first will allow us to determine if the changes are in scope.
3. If the changes are agreed, then you can go ahead and set up a development environment (see [Setting Up Your Development Environment](#setting-up-your-development-environment) below).
4. Once you have commits ready to be shared, initiate a draft Pull Request with an initial version of your implementation and request feedback. It's advisable not to wait until the feature is fully completed.
5. Ensure that the Pull Request adheres to the correct naming conventions:wh 

## Setting Up a Development Environment

1. Fork on GitHub:
    Go to the [repository's page](https://github.com/aurelio-labs/semantic-router) on GitHub: 
    Click the "Fork" button in the top-right corner of the page.
2. Clone Your Fork:
    After forking, you'll be taken to your new fork of the repository on GitHub. Copy the URL of your fork from the address bar or by clicking the "Code" button and copying the URL under "Clone with HTTPS" or "Clone with SSH".
    Open your terminal or command prompt.
    Use the git clone command followed by the URL you copied to clone the repository to your local machine. Replace &lt;Your-Fork-URL&gt; with the URL of your fork
    ```
    git clone https://github.com/<your-gh-username>/<semantic-route>.git
    ```
4. Ensure you have poetry installed: `pip install poetry`.
3. Then navigate to the cloned folder, create a virtualenv, and install via poetry (which defaults to an editable installation):
    ```
    # Move into the cloned folder
    cd haystack/

    # Create a virtual environment
    python3 -m venv venv

    # Activate the environment
    source venv/bin/activate

    # Install via poetry
    poetry install
    ```
    
