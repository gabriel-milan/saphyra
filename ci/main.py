#
#   Call ALL your CI scripts from here
#

from test_import import test_import
# from test_callbacks import test_sp_callback
from test_sp_metric import test as test_metric


if __name__ == '__main__':
    print (" ===> BEGIN CI TEST <=== ")
    print (" ==> Import saphyra test")
    test_import()
    print (" ==> Testing tuning")
    test_tuning()
