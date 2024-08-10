from string import ascii_lowercase

# get all the names as a list
# input data is stored in a file, with one name per line
def get_names(filenanme:str)->list:
    names = []
    with open(filenanme,mode="r") as f:
        # strip \n and empty lines
        names = [line.strip().lower() for line in f if line.strip()]
    return names

# convert letter to integer
# letters are a to z. 
# Special token "." is used for beginning and end of sequence
# . is encoded as 0
# a as 1, etc...
def stoi(letter:str)->int:
    if letter == ".":
        return 0
    return ord(letter) -96

# convert integer to letter
# 0 -> .
# 1 -> a, 2->b, etc..
def itos(i:int) ->str:
    if i == 0:
        return "."
    return chr(i+96)