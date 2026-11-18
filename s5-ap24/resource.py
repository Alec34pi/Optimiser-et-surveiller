RUSAGE_SELF = 0
def getrusage(who):
    class DummyRusage:
        ru_maxrss = 0
    return DummyRusage()