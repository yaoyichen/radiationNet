import os


class FileUtils(object):
    def __init__(self):
        super().__init__()
    pass

    @staticmethod
    def makedir(dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        f = open(os.path.join(dirs, filename), "a")
        f.close()


def main():
    pass


if __name__ == "__main__":
    main()
