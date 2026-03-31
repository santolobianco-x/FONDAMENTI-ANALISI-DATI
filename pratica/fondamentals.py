#QUESTA È UNA PICCOLA VIEW DELLE COSE BASILARI IN PYTHON
#!!!NON ESISTONO ; 
#-si rimuovono dichiarazione delle variabili tradizionali es(C). int a = 5;
print("FIRST POINT")
a = 5
print("The value of a is {}."\
      .format(a))# \serve se si vuole mandare a capo all'interno di print
aa = 57.89
print("The value of aa is {:0.2f}."\
      .format(aa))
aaa = 'c'
print("The value of aaa is {}"\
      .format(aaa))
aaaa = "marco castello"
print(a,aa,aaa,aaaa)
#IL DOMINIO DELLE VARIABILE PUÒ CAMBIARE DATO CHE È DINAMICO
a = "ciao"
print("The value of a is {}"\
      .format(a))

#/\PER VERIFICARE IL DOMINIO DI APPARTENZA BASTA UTILIZZARE type
print("The type of a is {}"\
      .format(type(a)))
#/\PER IL CASTIN DELLE VARIABILI SI USA QUELLO TRADIZIONALE
print("Original value of aa is {} \nCasted value of aa is {}"\
      .format(aa,int(aa)))

#/\PER INPUT DA TERMINALE
aaaaa = input("Enter your favorite song | your data of birth | your amount\n<")
print("The value of aaaaa is {}"\
      .format(aaaaa))

print()
print()
print()

#-liste semplici da gestire
print("SECOND POINT")
b = []
print("b is composed by {}"\
      .format(b))
#/\PER INSERIRE NUOVI ELEMENTI SI USA append
b.append(5)
b.append("delitto e castigo")
b.append(3.14)
#/\SI PUÒ ANCHE UTILIZZARE insert, MA SI DEVE SPECIFICARE L'INDEX
b.insert(1,"ciao")

print("b is composed by {}".format(b))

#/\PER RIMUOVERE UN ELEMENTO SI UTILIZZA LA FUNZIONE pop
#pop(x) CON x INDICE

print("Value removed in b by pop() is {}"
      .format(b.pop(0)))

#/\PER RIMUOVERE UN ELEMENTO/INTERVALLO DI ELEMENTI SI UTILIZZA LA FUNZIONE del
#PER L'INTERVALLO, GLI ELEMENTI RIMOSSI € [x,y[
print("Value removed in b from index 0 to index 2")
del b[0:2:1] #SLICING [start:stop:step]

print("b is composed by {}".format(b))

bb = (1,2,3) #tuple-> liste a lunghezza fissa e costanti

print()
print()
print()


#-stringhe
print("THIRD POINT")
c = "ciao" #su una singola linea
cc = """ciao sono
santo scritto su multilinea""" #su multilinea


#VARIE FUNZIONI UTILI PER LE STRINGHE LETTERALI
print("any function applied to strings")
print(c.upper())
print(c.capitalize())
print("HELLO".lower())


#/\PER CONCATENARE STRINGHE SI UTILIZZA join
#LA STRINGA ANTECEDENTE A join È IL SEPARATORE
print()
print("String obtained by join:")
ccc = "_".join([c,cc,"FINE"]) 
print(ccc)


#/\PER DIVIDERE STRINGHE IN TOKEN SI UTILIZZA split
print()
print("Strings obtained by split:")
for x in ccc.split("_"):
    print(x)

print()
print()
print()



#-dizionari
#MOLTO EAZY, SONO DELLE LISTE CHE FUNZIONANO [chiave:valore]
#DOVE LA chiave PUÒ ASSUMERE QUALSIASI VALORE
print("FOURTH POINT")
d = dict()
d['ciao'] = 5
d[1] = '??'

#così si controllano le chiavi
if 1 in d:
    print(True)

if 5 in d.values():
    print(True)

d[(5,"ciao",23)] = 65

searchedkey = [k for k, v in d.items() if v == 65]
print("Key associated to 65 is {}".format(searchedkey))


print()
print()
print()


#-set
#FUNGONO DA INSIEMI, QUINDI NON POSSONO AVERE VALORI DUPLICATI ALL'INTERNO
#SI POSSONO APPLICARE OPERAZIONI INSIEMESTICHE
#POSSONO CONTENERE QUALSIASI ELEMENTO A MENO DI LISTE SET DIZIONARI
#GLI ELEMENTI ALL'INTERNO DEL SET SONO IMMUTABILI NEL TEMPO
print("FIFTH POINT")
e = {1,2,3}
ee = set()
ee.add(3)
ee.add(4)
ee.add(5)
print("The set e is composed by {}".format(e))
print("The set ee is composed by {}".format(ee))
print("Union of e and ee {}" .format(e|ee))
print("Intersection of e and ee {}" .format(e&ee))
print("Difference of e and ee {}".format(e-ee))
print("Difference of ee and e {}".format(ee-e))
if e <= ee:
    print("e is a subset of ee")
else:
    print("e isn't a subset of ee")

print()
print()
print()



#-functions
def func(x):
    return x+x+x

f = func(55)

#FUNZIONI ANONIME, SI USANO IN PARTICOLARE PER PASSARLE AD ALTRE FUNZIONI
ff = lambda x:x*2
fff = lambda x,y:(x,+1,y+1000)

#print(fff(4,5)) to reclaim



#-map
#map PERMETTE DI APPLICARE UNA FUNZIONA AD UN'INTERA LISTA
print("SIXTH POINT")
def square(x):
    return x*x

g = [1,2,3,4,5,6,7,8,9,10]
print("g before map+func is {}".format(g))
g = list(map(square,g))
print("g after map+func is {}".format(g))


print()
print()
print()



#-filter
#filter PERMETTE DI APPLICARE UN FILTRO AD UNA LISTA
print("SEVENTH POINT")
def isodd(x):
    if x%2 == 1:
        return x


h = [1,2,3,4,5,6,7,8,9,10]
print("h before applied filter of odd is {}".format(h))
h = list(filter(isodd,h))
print("h after applied filter of odd  is {}".format(h))


print()
print()
print()

