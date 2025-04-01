# Testing guidelines for API and Machine Learning Projects

This document provides best practices and guidelines for testing your project.
It will help you to ensure that your code is reliable, robust, and
maintainable, it also provides a comprehensive understanding of how to
structure, write, and maintain effective tests to ensure your code remains
reliable and robust.

## What to test your software?

Testing is a crucial part of software development. Well-written tests ensure
that your code fulfills your requirements and behaves as expected. When a
project grows, it becomes harder to add new features or fix bugs without
breaking existing functionalities. As manual testing is time-consuming and
error-prone, adding tests will ensure that you can make changes to your
codebase without and rapidly check everything will working as expected.

## Why Use pytest?

Most of machine learning projects are written in Python, therefore, we
it is natural to use a high featured testing framework like `pytest`.
It is easy to use, supports fixtures for managing test setup and teardown,
and has a rich ecosystem of plugins for extending functionality.

## Project Structure

Organize your project to separate the test code from the main codebase.
A typical structure for an API ML code might look like this:

```
thermal-anomaly-segmenter/
│
├── .sqa
│   ├── config.yml          <-- Software Quality Assurance configuration
│   ├── docker-compose.yml  <-- Compose file for integration configuration
│   └── dockerfile.py       <-- Instructions to build testing container
|
├── api
│   ├── __init__.py         <-- API importable objects
│   └── ...                 <-- See base README.md for more info
│
├── (<your_model_source>)*  # Optional
│   ├── __init__.py         <-- Model importable objects
│   ├── data_processing.py  <-- Data preprocessing functions
│   ├── model_training.py   <-- Model training functions
│   └── model_prediction.py <-- Model prediction functions
│
├── tests
│   ├── configurations      <-- Configuration files to test
|   ├── data                <-- Data to test and use for testing
│   ├── test_metadata       <-- Tests for metadata endpoints
│   ├── test_predictions    <-- Tests for prediction endpoints
│   ├── test_training       <-- Tests for training endpoints
│   └── conftest.py         <-- Pytest fixtures
│
├── Jenkinsfile             <-- CICD pipeline configuration for Jenkins
├── pyproject.toml          <-- Project configuration file
├── requirements-test.txt   <-- Testing requirements
├── tox.ini                 <-- Standardization testing config
└── ...                     <-- See base README.md for more info
```

This structure contains a lot of components, but we will introduce them
gradually in the following sections. Model source\* is optional for this
template. As its main purpose is to provide a model API, in some cases
the model source is not included in the project but installed from a package
or as a submodule.

## Best Practices

### 1. Test Data Management

- **Use Small, Representative Datasets**:
  Create small datasets that represent the variety of cases your model will
  encounter. For example a few Mb model or a few images to use as input to
  your load, predict and train functions.
- **Avoid Using Real Data and Used Secrets**:
  Use synthetic or anonymized data for testing to ensure privacy and
  reproducibility. When you require some credential or passwords, do not use
  the same you will use in production, use different ones for testing.

### 2. Test Coverage

- **Cover All Critical Paths**:
  Test writing is time-consuming, so focus on testing the critical paths,
  reach the highest coverage possible, but keep in mind that 100% coverage
  will not guarantee that your code is bug-free.
- **Edge Cases and Invalid Inputs**:
  Test some invalid inputs to ensure your code handles them gracefully.

### 3. Fixture Usage

- **Leverage Fixtures**:
  Use fixtures to manage setup and teardown of test resources (e.g., datasets,
  model instances). Set the highest scope as possible to reduce testing time.
- **Modular Fixtures**:
  Create reusable fixtures in your `conftest.py` files to avoid code
  duplication. Divide them between folders to simplify the maintenance.
- **Error vs Failures**:
  Use your fixtures to execute code and your test files to assert the results.
  This will improve the readability of your tests and make it easier to
  understand what is failing. An error means your code failed, a failure means
  your code returned an unexpected result.

### 4. Test only what you own

- **Isolate Tests**:
  Only integration tests should test the interaction between different
  components. When doing unit tests, test only the code you own, not the
  libraries you are using. In most cases you do not need to test a real model
  but is enough to mock one with its `predict` and `fit` methods.
- **Mock using Specs**:
  Utilize libraries such as `unittest.mock` to create mock objects.
  When you mock an object or function, use `autospec=True` to ensure that
  the test fails if you call the mock with unexpected arguments. I recommend
  to read
  [Python Unittest Mock Autospeccing](https://docs.python.org/3/library/unittest.mock.html#autospeccing).

### 5. Control your test environment

- **Include tox for validation**:
  You do not need to run always tests using tox, but before pushing your code
  to the repository, you should run tox to ensure that your code is working
  in different environments and machines or Python versions.
- **Use src folder when developing an installable**:
  If your code is planned to be installed, use a `src` folder to store your
  code. Python adds your working directory to the path, and `import` command
  might succeed on your development machine but fail when installed.
- **Split test and software requirements**:
  Keep your test requirements separate from your software requirements.
  This will make it easier to manage dependencies and ensure that your
  production code does not include testing libraries.
- **Configure pytest at pyproject.toml**:
  Use `pyproject.toml` to configure `pytest` and plugins. This will make your
  project more maintainable and easier to share with others.

### 6. Test code quality and security

- **Check code quality and style**:
  Use `flake8` or other tools to ensure your code follows the programming
  recommendations and style guides. This will make your code more readable
  and maintainable.
- **Check for security vulnerabilities**:
  Use `bandit` or other tools to check for security vulnerabilities in your
  code. This will help you to identify and fix potential security issues.

## Writing your Tests

All your tests should be placed in the `tests` folder and start with `test_`
meanwhile you should try to locate your fixtures in the `conftest.py` file.
The command `pytest` automatically discovers and run all files that match
this pattern, so it should be easy to start and edit the project tests.

Specific to this template, you have a folder `api` with the code expected to
be tested. Some projects include a "model source code", which contains the
code to train and predict the model. This template includes basic tests for
testing the `api` methods, which expect to make use of your model code and
therefore test it indirectly.

As explained in the best practices, you should restrict the tests in your
repository to the code you own. This means that you should not test an
external library of model, specially if that one already includes tests
of its own. For example, you might be building an API for `yolov` model,
in that case, you are not expected to have `yolov` source code in the
repository as it is already tested in its own repository.

## Running your Tests

To run your tests in your current environment, simply execute:

```bash
python -m pytest tests
```

Some IDEs like Visual Studio Code have built-in support for pytest.

For more advanced testing, you can use `tox` to run tests in multiple
environments. To run tests with `tox`, execute:

```bash
python -m tox
```

Tox will run tests in multiple environments, such as different Python versions
in this template, tox include custom environments to test your code quality
and security with `flake8` and `bandit`.

## Continuous Integration

CICD is the last step on the automation of the testing process. This template
uses Jenkins and [SQAaaS](https://sqaaas.eosc-synergy.eu/#/) to automate the
testing process. You can use the provided `.sqa` folder to configure your own
testing pipeline for project.

Jenkinsfile in this project is configured to use the Jenkins server and tools at
[jenkins.services.ai4os.eu](https://jenkins.services.ai4os.eu/job/AI4OS-hub/).
We do not recommend to edit the Jenkinsfile, but you add your configuration
files to the `.sqa` folder to customize your testing pipeline.
