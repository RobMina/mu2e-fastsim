def pytest_addoption(parser):
    parser.addoption('--offline', help='Read-only path to exact audited Offline source snapshot')
