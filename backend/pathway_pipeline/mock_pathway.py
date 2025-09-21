# Mock Pathway module for development without Pathway dependency

class MockTable:
    def __init__(self, data=None):
        self.data = data or []
    
    def select(self, *args, **kwargs):
        return MockTable(self.data)
    
    def filter(self, *args, **kwargs):
        return MockTable(self.data)
    
    def groupby(self, *args, **kwargs):
        return MockTable(self.data)
    
    def reduce(self, *args, **kwargs):
        return MockTable(self.data)

class MockConnector:
    @staticmethod
    def csv(*args, **kwargs):
        return MockTable()
    
    @staticmethod
    def jsonlines(*args, **kwargs):
        return MockTable()

class MockPathway:
    Table = MockTable
    io = MockConnector()
    
    @staticmethod
    def run(*args, **kwargs):
        pass

# Create mock pathway instance
pw = MockPathway()