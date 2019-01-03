from setuptools import setup, find_packages

requires = [
    "pyramid",
    "numpy",
    "choix",
    "msgpack",
    "blessed",
    "python-magic",
    "torch",
    "pudb",
    "watchdog",
    "psutil",
    "pytest",
    "hypothesis",
    "pytest-cov",
    "pytest-cache",
    "waitress",
]

setup(
    name="app",
    install_requires=requires,
    python_requires=">=3.6",
    packages=find_packages(exclude=["test*"]),
    entry_points="""\
        [paste.app_factory]
        main = web.app:make_app
    """,
)
