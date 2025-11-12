# Claude Guidelines for Python Projects (with uv)

This template provides a foundation for developing Python applications with Claude Code assistance. It combines architecture, testing, and documentation best practices with a **uv-first** workflow for packaging and tooling.

## Table of Contents

1. Project Overview
2. Core Architecture Principles
   2.1 Error Handling & Exception Management
   2.2 Type Hints & Static Typing
   2.3 Async/Await & Concurrency
   2.4 Configuration & Dependency Injection
3. File & Directory Structure
4. Code Style & Standards
   4.1 Documentation
   4.2 Logging Standards
   4.3 Testing Requirements
5. Common Patterns & Anti-Patterns
6. Development Workflow
   6.1 Feature Development
   6.2 Code Review Checklist
7. Performance Considerations
8. Security & Privacy
9. Package & Tooling Management (uv — Required)
   9.1 Package Management with uv
   9.2 Running Python & Tools with uv
   9.3 Managing Scripts with PEP 723 Inline Metadata
   9.4 Essential Tools
   9.5 Development Setup (with uv)
   9.6 IDE Configuration
10. Common Dependencies
11. Example Prompts for Claude

---

## 1) Project Overview

This template provides a foundation for developing Python applications with Claude assistance. It includes best practices for project structure, code organization, testing, and documentation that align with Python idioms and ecosystem conventions. **All dependency and tool execution is managed with `uv`.**

---

## 2) Core Architecture Principles

### 2.1 Error Handling & Exception Management

* **Use specific exceptions**: Prefer specific exception types over generic `Exception`
* **EAFP principle**: “Easier to Ask for Forgiveness than Permission” — use `try/except` over pre-checks
* **Context managers**: Use `with` for resource management
* **Custom exceptions**: Create domain-specific exception hierarchies

```python
from contextlib import contextmanager
from typing import Iterator

class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    pass

class ValidationError(DataProcessingError):
    """Raised when data validation fails."""
    pass

@contextmanager
def process_file(path: str) -> Iterator[str]:
    """Process a file with proper resource management."""
    try:
        with open(path, 'r') as f:
            yield f.read()
    except FileNotFoundError:
        raise DataProcessingError(f"File not found: {path}")
    except IOError as e:
        raise DataProcessingError(f"Failed to read file: {path}") from e
```

### 2.2 Type Hints & Static Typing

* **Use type hints** in function signatures
* **Generics** via `typing`
* **Type checking**: run `mypy` (via `uv run mypy …`)
* **Runtime validation**: consider `pydantic`

```python
from typing import List, Dict, Optional, Union, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    value: Optional[T] = None
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None

def process_data(
    data: List[Dict[str, Union[str, int]]], 
    filter_key: str
) -> Result[List[str]]:
    try:
        filtered = [
            str(item.get(filter_key, '')) 
            for item in data 
            if filter_key in item
        ]
        return Result(value=filtered)
    except Exception as e:
        return Result(error=str(e))
```

### 2.3 Async/Await & Concurrency

* **asyncio** for I/O-bound operations
* **threading** for certain concurrency with GIL caveats
* **multiprocessing** for CPU-bound parallelism
* **concurrent.futures** as a high-level interface

```python
import asyncio
import aiohttp
from typing import List, Dict

async def fetch_data(session: aiohttp.ClientSession, url: str) -> Dict:
    async with session.get(url) as response:
        return await response.json()

async def fetch_multiple(urls: List[str]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### 2.4 Configuration & Dependency Injection

* **Env vars**: `.env` + `pydantic` settings
* **DI**: explicit passing or a DI lib (e.g., `injector`)
* **Feature flags**: env or config files

```python
from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "MyApp"
    debug: bool = Field(False, env="DEBUG")
    database_url: str = Field(..., env="DATABASE_URL")
    api_key: str = Field(..., env="API_KEY")
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

---

## 3) File & Directory Structure

### Standard Layout

```
python-project/
├──package_name/       # Main package
│   ├── __init__.py
│   ├── __main__.py     # Entry point for -m execution
│   ├── main.py
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── api/
├── tests/                  # pytest tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── test_*.py
├── docs/                   # Sphinx or MkDocs
├── scripts/                # Utility scripts
├── data/                   # (optional) datasets
├── notebooks/              # (optional) Jupyter notebooks
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml          # Project & tool config (managed with uv)
├── uv.lock                 # Resolved dependency lock (managed with uv)
├── Dockerfile              # (optional) container config
└── README.md
```

> Note: We **do not** maintain `requirements/` or `requirements.txt` files. Use `uv` and `pyproject.toml` exclusively.

**Naming conventions**

* Python modules: `snake_case.py`
* Tests: `test_*.py`
* Constants-only modules: UPPER_CASE is fine
* Private modules: prefix with `_`

---

## 4) Code Style & Standards

### 4.1 Documentation

* **Docstrings**: Google- or NumPy-style, consistently
* **Type hints**: required in signatures
* **Module-level docstrings** at top
* **Examples**: include in docstrings when helpful

```python
"""Module for user authentication and authorization."""
from typing import Optional, Dict, Any

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user with username and password.

    Args:
        username: The user's username.
        password: The user's password (will be hashed).
    Returns:
        Dict with user info if success, else None.
    Raises:
        ValueError: If username or password is empty.
    """
    if not username or not password:
        raise ValueError("Username and password are required")
    return None
```

### 4.2 Logging Standards

* **Structured logging** or stdlib `logging` with formatters
* Proper **log levels**
* Include contextual fields
* Avoid heavy computation in log calls

```python
import logging
from functools import wraps
from typing import Callable, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_execution(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Executing {func.__name__}", extra={
            'function': func.__name__,
            'module': func.__module__
        })
        try:
            result = func(*args, **kwargs)
            logger.info(f"Successfully executed {func.__name__}")
            return result
        except Exception as e:
            logger.error(
                f"Error in {func.__name__}: {str(e)}",
                exc_info=True,
                extra={'function': func.__name__}
            )
            raise
    return wrapper
```

### 4.3 Testing Requirements

* **pytest** as the primary framework
* **Coverage** target: ≥ 80%
* **Fixtures** for setup/teardown
* **Mocking**: `unittest.mock` or `pytest-mock`
* **Property testing**: `hypothesis` where appropriate

```python
# tests/test_user_service.py
import pytest
from unittest.mock import patch
from hypothesis import given, strategies as st

from src.package_name.services.user import UserService
from src.package_name.models.user import User

@pytest.fixture
def user_service():
    return UserService()

@pytest.fixture
def mock_database():
    with patch('src.package_name.services.user.get_db_connection') as mock:
        yield mock

def test_create_user(user_service, mock_database):
    mock_database.return_value.save.return_value = True
    user = user_service.create_user(name="Alice", email="alice@example.com")
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    mock_database.return_value.save.assert_called_once()

@given(name=st.text(min_size=1, max_size=50), email=st.emails())
def test_user_validation(name, email):
    user = User(name=name, email=email)
    assert user.is_valid()
```

---

## 5) Common Patterns & Anti-Patterns

**Do**

* Use context managers
* Follow PEP 8
* Use type hints
* Write docstrings
* Use virtualized environments (handled by uv)
* Implement proper logging
* Write tests alongside code
* Use comprehensions judiciously
* Handle exceptions specifically

**Don’t**

* Use mutable default args
* `from module import *`
* Ignore exceptions
* Overuse globals
* Write unreadable comprehensions
* Modify a list while iterating
* Use `eval()`/`exec()` on user input
* Hardcode secrets

---

## 6) Development Workflow

### 6.1 Feature Development

1. **Design first**: interfaces & data structures
2. **Write tests** (TDD or close)
3. **Implement incrementally**
4. **Document** (docstrings & docs)
5. **Refactor** with tests green
6. **Review** before submitting

### 6.2 Code Review Checklist

* [ ] PEP 8 & style guide
* [ ] Type hints present
* [ ] Comprehensive tests
* [ ] Clear documentation
* [ ] Lint/type checks pass (`uv run …`)
* [ ] Appropriate exception handling
* [ ] Security best practices
* [ ] Perf considerations addressed

---

## 7) Performance Considerations

* **Profiling**: `cProfile`, `line_profiler`
* **Caching**: `functools.lru_cache`
* **Lazy evaluation**: generators
* **Vectorization**: NumPy/Pandas
* **Async I/O** where relevant

```python
from functools import lru_cache
from typing import Iterator, List
import time

@lru_cache(maxsize=128)
def expensive_calculation(n: int) -> int:
    time.sleep(0.1)
    return n ** 2

def process_large_dataset(data: List[int]) -> Iterator[int]:
    for item in data:
        if item % 2 == 0:
            yield expensive_calculation(item)

results = list(process_large_dataset(range(1000)))
```

---

## 8) Security & Privacy

* **Validate & sanitize** all inputs
* **Parameterized queries** to prevent SQL injection
* **Secrets** via env/config (never hardcode)
* **Audit dependencies** regularly
* **Auth** via established libs (e.g., `passlib`)

```python
import secrets
import hashlib
from typing import Tuple

def generate_secure_token() -> str:
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> Tuple[str, str]:
    salt = secrets.token_hex(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return salt, pwdhash.hex()

def verify_password(password: str, salt: str, pwdhash: str) -> bool:
    new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return new_hash.hex() == pwdhash
```

---

## 9) Package & Tooling Management (uv — Required)

**Use `uv` exclusively** for Python package management in this project. Do **not** use pip, pip-tools, poetry, or conda directly.

### 9.1 Package Management with uv

* Add dependency:
  `uv add <package>`
* Remove dependency:
  `uv remove <package>`
* Sync (install/resolve lock):
  `uv sync`

### 9.2 Running Python & Tools with uv

* Run a script:
  `uv run path/to/script.py`
* Run a module:
  `uv run -m package_name`
* Run tools (examples):

  * `uv run pytest`
  * `uv run ruff`
  * `uv run mypy`
  * `uv run sphinx-build -b html docs docs/_build`

### 9.3 Managing Scripts with PEP 723 Inline Metadata

* Run a script that declares deps inline at the top:
  `uv run script.py`
* Modify inline deps via CLI:

  * `uv add <package> --script script.py`
  * `uv remove <package> --script script.py`

### 9.4 Essential Tools

* **ruff**: Linting + formatting (preferred over flake8/black combo if desired)
* **mypy**: Type checking
* **pytest** + **pytest-cov**: Testing and coverage
* **pre-commit**: Git hooks
* **sphinx** or **mkdocs**: Documentation

> If you use `flake8`/`black` instead of `ruff`, run them via `uv run black …`, `uv run flake8 …`.

### 9.5 Development Setup (with uv)

```bash
# 1) Sync and create the environment from pyproject.toml
uv sync

# 2) Install pre-commit hooks
uv run pre-commit install

# 3) Run all checks
uv run ruff check src tests
uv run mypy src
uv run pytest --cov=src
```

### 9.6 IDE Configuration

* **VS Code**: Python + Pylance; point interpreter to `.venv` created by `uv` (or uv-managed env)
* **PyCharm**: Use the uv-created environment as the project interpreter
* Maintain configs in `pyproject.toml`, `.editorconfig`, and optional `setup.cfg` as needed for tools

---

## 10) Common Dependencies

**Core Libraries**

* `httpx`/`requests`: HTTP clients
* `pydantic`: Validation & settings
* `typer`/`click`: CLI frameworks
* `fastapi`/`flask`: Web frameworks
* `sqlalchemy`: ORM
* `celery`/`rq`: Task queues

**Testing**

* `pytest`, `pytest-cov`, `pytest-mock`, `hypothesis`, `factory-boy`, `responses`

**Dev Tools**

* `ruff` (or `black` + `isort` + `flake8`)
* `mypy`
* `pre-commit`
* `tox` (optional; can also run via `uv run tox`)

---

## 11) Example Prompts for Claude

### Implementing New Features

```
Implement a REST API endpoint for user management using FastAPI with:
- CRUD operations for users
- JWT authentication
- Input validation with Pydantic models
- Proper error handling and status codes
- Comprehensive tests using pytest
- OpenAPI documentation
```

### Fixing Issues

```
Fix the async database connection pool issue in src/db/connection.py
where connections are not being properly released during concurrent requests.
Ensure proper context manager usage and add tests to verify the fix.
```

### Refactoring

```
Refactor the data processing module to use pandas DataFrames instead of raw dictionaries.
Maintain backward compatibility, improve performance for large datasets, and add type hints.
Include benchmarks comparing the old and new implementations.
```

### Performance Optimization

```
Optimize the image processing pipeline in src/imaging/processor.py for better performance.
Profile the current implementation, identify bottlenecks, and implement improvements using
multiprocessing or async I/O where appropriate. Target a 50% performance improvement.
```