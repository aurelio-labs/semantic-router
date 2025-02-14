# Contributing to the Semantic Router

The Aurelio Team welcome and encourage any contributions to the Semantic Router, big or small. Please feel free to contribute to new features, bug fixes, or documentation. We're always eager to hear your suggestions.

Please follow these guidelines when making a contribution:
1. **Check for Existing Issues:** Before making any changes, [check here for related issues](https://github.com/aurelio-labs/semantic-router/issues).
2. **Run Your Changes by Us!** If no related issue exists yet, please create one and suggest your changes. Checking in with the team first will allow us to determine if the changes are in scope.
3. **Set Up Development Environment** If the changes are agreed, then you can go ahead and set up a development environment (see [Setting Up Your Development Environment](#setting-up-your-development-environment) below).
4. **Create an Early Draft Pull Request** Once you have commits ready to be shared, initiate a draft Pull Request with an initial version of your implementation and request feedback. It's advisable not to wait until the feature is fully completed.
5. **Ensure that All Pull Request Checks Pass** There are Pull Request checks that need to be satifisfied before the changes can be merged. These appear towards the bottom of the Pull Request webpage on GitHub, and include:
    - Ensure that the Pull Request title is prepended with a [valid type](https://flank.github.io/flank/pr_titles/). E.g. `feat: My New Feature`.
    - Run linting (and fix any issues that are flagged) by:
        - Navigating to /semantic-router.
        - Running `make lint` to fix linting issues.
        - Running `black .` to fix `black` linting issues.
        - Running `ruff check . --fix` to fix `ruff` linting issues (where possible, others may need manual changes).
        - Running `mypy .` and then fixing any of the issues that are raised.
        - Confirming the linters pass using `make lint` again.
    - Ensure that, for any new code, new [PyTests are written](https://github.com/aurelio-labs/semantic-router/tree/main/tests/unit). If any code is removed, then ensure that corresponding PyTests are also removed. Finally, ensure that all remaining PyTests pass using `pytest ./tests` (to avoid integration tests you can run `pytest ./tests/unit`.
    - Codecov checks will inform you if any code is not covered by PyTests upon creating the PR. You should aim to cover new code with PyTests.

> **Feedback and Discussion:**
While we encourage you to initiate a draft Pull Request early to get feedback on your implementation, we also highly value discussions and questions. If you're unsure about any aspect of your contribution or need clarification on the project's direction, please don't hesitate to use the [Issues section](https://github.com/aurelio-labs/semantic-router/issues) of our repository. Engaging in discussions or asking questions before starting your work can help ensure that your efforts align well with the project's goals and existing work.

# Setting Up Your Development Environment

1. Fork on GitHub:
    Go to the [repository's page](https://github.com/aurelio-labs/semantic-router) on GitHub: 
    Click the "Fork" button in the top-right corner of the page.

2. Clone Your Fork:
    After forking, you'll be taken to your new fork of the repository on GitHub. Copy the URL of your fork from the address bar or by clicking the "Code" button and copying the URL under "Clone with HTTPS" or "Clone with SSH".
    Open your terminal or command prompt.
    Use the git clone command followed by the URL you copied to clone the repository to your local machine. Replace `https://github.com/<your-gh-username>/<semantic-router>.git` with the URL of your fork:
    ```
    git clone https://github.com/<your-gh-username>/<semantic-router>.git
    ```

3. Ensure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/), for macos and linux use `curl -LsSf https://astral.sh/uv/install.sh | sh`.

4. Then navigate to the cloned folder, create a virtualenv, and install via `uv`:
    ```
    # Move into the cloned folder
    cd semantic-router/

    # Create a virtual environment
    uv venv --python 3.13

    # Activate the environment
    source .venv/bin/activate

    # Install via uv with all extras relevant to perform unit tests
    uv pip install -e .[dev]
    ```
