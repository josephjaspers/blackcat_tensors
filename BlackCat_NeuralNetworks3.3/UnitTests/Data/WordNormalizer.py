
import sys

def main(*args):
    valid_symbols = "abcdefghijklmnopqrstuvwxyz "

    for arg in args:
        file = open(arg, 'r')

        txt = file.read()
        txt = txt.lower()

        txt = "".join([c for c in txt if c in valid_symbols])
        txt = txt.split(' ')

        file = open("fixed_" + arg, 'w')
        file.write("std::vector<std::string> " + arg[0:-4]+" {")
        for words in txt:
            if words != "," and words != "":
                file.write('"' + words + '",')
                file.write('\n')

        file.write(" };")
        file.close()
#
# if __name__ == "__main__":
#     main(sys.argv[1:])


main("TheRaven.txt", "AngelOfTheOdd.txt", "BlackCat.txt", "TheCaskOfAmontillado.txt", "ThePit.txt")

