import Levenshtein as Lev
from string import ascii_uppercase

def wer(s1, s2):

    s1 =s1.upper()
    s2 =s2.upper()
    b = set(s1.split() + s2.split())
    
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Lev.distance(''.join(w1), ''.join(w2))/float(len(s2.split()))

def indices_to_string(indices):
	
    space_token = ' '
    end_token = '>'
    blank_token = '%'
    apos_token = '\''
        
    alphabet = list(ascii_uppercase) + [space_token, apos_token, blank_token, end_token] 

    sentence = ''
    for idx in indices:
        sentence += alphabet[idx]
    
    return sentence