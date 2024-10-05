import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split #bus reikalinga mokymo daliai
#Stulpelių pavadinimai vėžio duomenų apsirašau, plius nuskaitymas
stulpeliai_vezio = ['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
Krūties_vėžys = r"C:\Users\micke\Desktop\Duomenų mokslas 2022-2026\3 Metai\2024 ruduo\Dirbtinis intelektas\Projektas nr. 2\breast+cancer+wisconsin+original\breast-cancer-wisconsin.data"
Krūties_vėžys_2 = pd.read_csv(Krūties_vėžys, header=None, names=stulpeliai_vezio)
#pakeitimas tuščių reikšmių
Krūties_vėžys_2.replace('?', pd.NA, inplace=True)#? pakeičiama į na
Krūties_vėžys_2.dropna(inplace=True)#šalinimas na
Krūties_vėžys_2.drop(columns=['ID'], inplace=True)#naikinimas ID stulpelio
Krūties_vėžys_2['Class'] = Krūties_vėžys_2['Class'].replace({2: 0, 4: 1})#keitimas klasių, jei 2 tai bus 0, jei 4 tai 1 
#Permaišymas visų duomenų,priskiriant nauja indeksa eilutei
Krūties_vėžys_2 = Krūties_vėžys_2.sample(frac=1).reset_index(drop=True)
print("Sutvarkyti duomenys:")
Krūties_vėžys_2.to_csv("tvarkyti_duomenys_krūties.csv", index=False)
#Irisų duomenys
iris_failas = r"C:\Users\micke\Desktop\Duomenų mokslas 2022-2026\3 Metai\2024 ruduo\Dirbtinis intelektas\Projektas nr. 2\iris\iris.data"
Stulpeliai_irisu = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_duomenys = pd.read_csv(iris_failas, header=None, names=Stulpeliai_irisu)
# Filtruoju pagal Versicolor ir Virginica irisų klases ir sukuriu kopiją
filtruota = iris_duomenys[iris_duomenys['class'].isin(['Iris-virginica','Iris-versicolor'])].copy()
filtruota.to_csv("tvarkyti_duomenys_irisu.csv", index=False)
# Pervadinu stulpelius, buvo galima labels, kaip rStudio pasidaryti, bet pamiršau, tad nemanau, kad blogai:)
filtruota.rename(columns={
    'class': 'Klasė',
    'sepal_length': 'Taurėlapio ilgis',
    'sepal_width': 'Taurėlapio plotis',
    'petal_length': 'Vainiklapio ilgis',
    'petal_width': 'Vainiklapio plotis'
}, inplace=True)

#Didinimas, iš duomenų filtruota išrenku klases reikiamas
versicolor = filtruota[filtruota['Klasė'] == 'Iris-versicolor']
virginica = filtruota[filtruota['Klasė'] == 'Iris-virginica']
#didinimo funkcija
def padidinti_su_triuksmu(duomenys, kiekis, triukšmas=0.01):#parametrai ir triukšmo lygis 
    # Didinimas duomenų, suskaičiuojama kiek kartų didinti jei pilnas rinkinys duomenų pagal kiekį, tai pridedame
    padidinti_duomenys = pd.concat([duomenys] * (kiekis // len(duomenys)) + [duomenys.sample(kiekis % len(duomenys))])
    #pd.concat funkcija apjungia duomenis
    # Pridedame triukšmą su for per visus 
    for reiksme in duomenys.select_dtypes(include=[float, int]).columns:
        padidinti_duomenys[reiksme] += np.random.normal(0, triukšmas, size=len(padidinti_duomenys))#pridedamas triukšmas
    
    return padidinti_duomenys
# Atvaizdavimas duomenų prieš/po didinimo, tikrinimas ar viskas tvarkoje
def atvaizduoti_duomenis(duomenys, pavadinimas, x, y, klases_stulpelis):
    # Paimame unikalių klasių reikšmes
    klases = duomenys[klases_stulpelis].unique()  
    
    for klase in klases:
        # Filtruojame duomenis pagal kiekvieną klasę
        klase_duomenys = duomenys[duomenys[klases_stulpelis] == klase]
        plt.scatter(klase_duomenys[x], klase_duomenys[y], label=klase)  
    legendos_pavadinimai = ["Įvairiaspalvis vilkdalgis (angl. Iris-versicolor)", "Mėlynasis vilkdalgis (angl. Iris-virginica)"]
    plt.title(pavadinimas)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(legendos_pavadinimai)
    plt.show()

atvaizduoti_duomenis(filtruota, "Pradiniai duomenys",'Taurėlapio ilgis', 'Taurėlapio plotis', 'Klasė')

#Didinimas duomenų iki 200 įrašų kiekvienai klasei
versicolor_padidinta_su_triuksmu = padidinti_su_triuksmu(versicolor, 200)
virginica_padidinta_su_triuksmu = padidinti_su_triuksmu(virginica, 200) 
#Apjungimas ir atvaizdavimas braižant 
iris_padidintas_su_triuksmu = pd.concat([virginica_padidinta_su_triuksmu,versicolor_padidinta_su_triuksmu]).sample(frac=1).reset_index(drop=True)
iris_padidintas_su_triuksmu.rename(columns={'class': 'Klasė'}, inplace=True)#kadangi blogai braižė, tad dar kartą pakoreguoju pavadinimus, PAMOKA, kad nereikia keisti tiesiogiai
iris_padidintas_su_triuksmu.rename(columns={'sepal_length': 'Taurėlapio ilgis', 'sepal_width': 'Taurėlapio plotis'}, inplace=True)
atvaizduoti_duomenis(iris_padidintas_su_triuksmu, "Duomenys su triukšmu", 'Taurėlapio ilgis', 'Taurėlapio plotis', 'Klasė')
# Išsaugome padidintus irisų duomenis
iris_padidintas_su_triuksmu.to_csv("padidinti_irisu_duomenys_su_triuksmu.csv", index=False)
# Vidurkių ir dispersijų apskaičiavimas
vidurkiai_pries = filtruota.drop(columns=['Klasė']).mean()
dispersijos_pries = filtruota.drop(columns=['Klasė']).var()
vidurkiai_po = iris_padidintas_su_triuksmu.drop(columns=['Klasė']).mean()
dispersijos_po = iris_padidintas_su_triuksmu.drop(columns=['Klasė']).var()
#print(filtruota.columns)
#print(iris_padidintas_su_triuksmu.columns)

# Palyginu vidurkius ir dispersijas
rezultatai = pd.DataFrame({
    "Vidurkis Prieš": vidurkiai_pries,
    "Vidurkis Po": vidurkiai_po,
    "Dispersija Prieš": dispersijos_pries,
    "Dispersija Po": dispersijos_po
})
print(rezultatai.to_string())


# Sigmoidinė aktyvacijos funkcija (f(ai))
def sigmoidine_funkcija(a_i):
    return 1 / (1 + np.exp(-a_i))
# Funkcija apskaičiuoti neuronų išėjimą (yi) remiantis sigmoidine funkcija
def prognozuoti(X_i, w, b):
    a_i = np.dot(X_i, w) + b  # skaičiuojama suma ai = Σ w_k * X_ik
    return sigmoidine_funkcija(a_i)  # sigmoidinė aktyvacijos funkcija f(ai)
# Klasifikavimo tikslumo skaičiavimas
def tikslumas(t_i, y_i):
    y_apvalinta = np.round(y_i)  # Suapvalinama iki 0 arba 1
    return np.mean(t_i == y_apvalinta)#vidutinė reikšmė grąžinama

# Funkcija neuronui mokyti naudojant stochastinį gradientinį nusileidimą
def Neurono_mokymas(X, t, greitis, epochos, E_min=0.01):  # X - mokymo duomenys, t - tikros klasės, eta - mokymosi greitis
     #E_min -paklaidos siekiamo tikslumo reikšmė
    np.random.seed(123)#seed'as
    m, n = X.shape  # m - įrašų skaičius, n - požymių skaičius( iš duomenų abu) , matrica mxn
    w = np.random.uniform(-1, 1, n)  # Atsitiktiniai pradiniai svoriai w_k, naudoju -1 ir 1 intervala
    b = np.random.uniform(-1, 1)  # Poslinkis(bias) w0

    totalError = float('inf')  # Paklaida nuo begalybės
    epocha=0
    totalError_VIS = []  # Paklaidos po kiekvienos epochos
    mokymo_tikslumas_viso = []  # Tikslumas po kiekvienos epochos

    while totalError > E_min and epocha < epochos:  # Kol paklaida didesnė už minimalią arba epochas
        totalError = 0  # Paklaidos suma epochai

        # Permaišymas 
        permaisymas = np.random.permutation(m)  # Sukuriame atsitiktinę seką indeksams
        X = X[permaisymas]  #
        t = t[permaisymas]  #

        for i in range(m):  # Eina per visus įrašus (Xi, ti)
            # Apskaičiuojamas neuronų išėjimas (yi)
            y_i = prognozuoti(X[i], w, b)  # yi = f(ai)

            # Klaida: (ti - yi), paskui kadratu
            klaida = t[i] - y_i  # ti - yi

            # Atnaujiname svorius w_k  formulę
            for k in range(n):  # Atnaujiname kiekvieną svorį
                w[k] += greitis * klaida * y_i * (1 - y_i) * X[i][k]  # w_k = w_k - η * (yi - ti) * yi * (1 - yi) * X_ik
            
            # Atnaujiname poslinkį ?????
            b += greitis * klaida * y_i * (1 - y_i)  # w0 = w0 - η * (yi - ti) * yi * (1 - yi)

            #  Error skaičiavimas
            totalError  += klaida**2  # Σ(ti - yi)^2

        # Apskaičiuojame paklaidą ir tikslumą po kiekvienos epochos
        prognozuoti_visi = prognozuoti(X, w, b)
        mokymo_tikslumas = tikslumas(t, prognozuoti_visi)
        
        totalError_VIS.append(totalError)#total error funkcijoje 
        mokymo_tikslumas_viso.append(mokymo_tikslumas)

        print(f'Epocha {epocha + 1}, Paklaidos: {totalError:.4f}, Tikslumas: {mokymo_tikslumas:.4f}')

        epocha+=1

    return w, b, totalError_VIS, mokymo_tikslumas_viso


def testavimas_neurono(X_test, t_test, w, b):
    # Prognozuojame neurono išėjimus testavimo duomenims
    y_prognozuota = prognozuoti(X_test, w, b)  # ats
    # Skaičiuojame tikslumą
    testavimo_tikslumas = tikslumas(t_test, y_prognozuota)  # klasifikavimo tikslumas
    # Skaičiuojame paklaidą E(W) 48skaidrė
    testavimo_paklaida = np.mean((t_test - y_prognozuota) ** 2)  #   
    # Grąžiname prognozuotas klases ir klaidas
    print(f'Testavimo paklaida: {testavimo_paklaida:.4f}, Klasifikavimo tikslumas: {testavimo_tikslumas:.4f}')

    return y_prognozuota, testavimo_paklaida, testavimo_tikslumas


def spausdinti_klases(y_prognozuota, t_test):
    print("Prognozuotos klasės ir tikrosios klasės:")
    for i in range(len(y_prognozuota)):
        print(f"Prognozuota klasė: {int(round(y_prognozuota[i]))}, Tikroji klasė: {int(t_test[i])}")
  

if __name__ == '__main__':
    greitis = float(input("Įveskite mokymo greitį intervale nuo 0 iki 1 pvz 0.02: "))
    epochos = int(input("Įveskite epochų skaičių: "))

    # Mokymo ir testavimo duomenys
    X_mok_krut_vezio, X_test_krut_vezio, t_mok_krut_vezio, t_test_krut_vezio = train_test_split(
        Krūties_vėžys_2.drop(columns=['Class']).values, 
        Krūties_vėžys_2['Class'].values, 
        test_size=0.2, random_state=1)

    iris_padidintas_su_triuksmu['Klasė'] = iris_padidintas_su_triuksmu['Klasė'].replace({'Iris-versicolor': 0, 'Iris-virginica': 1})

    # Mokymo ir testavimo duomenys
    X_mok_iris, X_test_iris, t_mok_iris, t_test_iris = train_test_split(
    iris_padidintas_su_triuksmu.drop(columns=['Klasė']).values, #panaikinu stulpelį kalsė, nes jis bus prognozuojamas
    iris_padidintas_su_triuksmu['Klasė'].values.astype(float),  # Konvertuojame į float
    test_size=0.2, random_state=1)

    X_mok_krut_vezio = np.array(X_mok_krut_vezio).astype(float)
    X_test_krut_vezio = np.array(X_test_krut_vezio).astype(float)
    X_mok_iris = np.array(X_mok_iris).astype(float)
    X_test_iris = np.array(X_test_iris).astype(float)

    np.random.seed(12)

    print("\nKrūties vėžio duomenys:")
    svoriai_kv, poslinkis_kv, mokymo_klaidos_kv, mokymo_tikslumas_kv = Neurono_mokymas(X_mok_krut_vezio, t_mok_krut_vezio, greitis, epochos)
    y_prognozuota_kv, testavimo_klaidos_kv, testavimo_tikslumas_kv = testavimas_neurono(X_test_krut_vezio, t_test_krut_vezio, svoriai_kv, poslinkis_kv)

    print(f"Galutiniai svoriai (Krūties vėžio duomenys): {svoriai_kv}")
    print(f"Galutinis poslinkis (bias): {poslinkis_kv}")
    print("\nKrūties vėžio testavimo duomenys (klasės):")
    spausdinti_klases(y_prognozuota_kv, t_test_krut_vezio)
    

    np.random.seed(123)

    # Mokome neuroną su irisų duomenimis
    print("\nIrisų duomenys:")
    svoriai_ir, poslinkis_ir, mokymo_klaidos_ir, mokymo_tikslumas_ir = Neurono_mokymas(X_mok_iris, t_mok_iris, greitis, epochos)
    y_prognozuota_ir, testavimo_klaidos_ir, testavimo_tikslumas_ir = testavimas_neurono(X_test_iris, t_test_iris, svoriai_ir, poslinkis_ir)

    print(f"Galutiniai svoriai (Irisų duomenys): {svoriai_ir}")
    print(f"Galutinis poslinkis (bias): {poslinkis_ir}")
    print("\nIrisų testavimo duomenys (klasės):")
    spausdinti_klases(y_prognozuota_ir, t_test_iris)

    # 1. Paklaidos priklausomybė nuo epochų skaičiaus:
    import matplotlib.pyplot as plt

    # Krūties vėžio duomenys - paklaidos grafikas
    epochos_kv = range(1, len(mokymo_klaidos_kv) + 1)
    plt.plot(epochos_kv, mokymo_klaidos_kv)
    plt.xlabel('Epochos')
    plt.ylabel('Paklaidos reikšmės')
    plt.title('Paklaidos priklausomybė nuo epochų skaičiaus (Krūties vėžio duomenys)')
    plt.show()

# Irisų duomenys - paklaidos grafikas
    epochos_ir = range(1, len(mokymo_klaidos_ir) + 1)
    plt.plot(epochos_ir, mokymo_klaidos_ir)
    plt.xlabel('Epochos')
    plt.ylabel('Paklaidos reikšmės')
    plt.title('Paklaidos priklausomybė nuo epochų skaičiaus (Irisų duomenys)')
    plt.show()
    # Krūties vėžio duomenys - tikslumo grafikas
    plt.plot(epochos_kv, mokymo_tikslumas_kv)
    plt.xlabel('Epochos')
    plt.ylabel('Klasifikavimo tikslumas')
    plt.title('Klasifikavimo tikslumas nuo epochų skaičiaus (Krūties vėžio duomenys)')
    plt.show()

    # Irisų duomenys - tikslumo grafikas
    plt.plot(epochos_ir, mokymo_tikslumas_ir)
    plt.xlabel('Epochos')
    plt.ylabel('Klasifikavimo tikslumas')
    plt.title('Klasifikavimo tikslumas nuo epochų skaičiaus (Irisų duomenys)')
    plt.show()
    #apjungimas 
    # Paklaidos grafikai
    plt.subplot(1, 2, 1)
    plt.plot(epochos_kv, mokymo_klaidos_kv, label='Krūties vėžio duomenys')
    plt.plot(epochos_ir, mokymo_klaidos_ir, label='Irisų duomenys')
    plt.xlabel('Epochos')
    plt.ylabel('Paklaidos reikšmės')
    plt.title('Paklaidos priklausomybė nuo epochų skaičiaus')
    plt.legend()

    # Tikslumo grafikai
    plt.subplot(1, 2, 2)
    plt.plot(epochos_kv, mokymo_tikslumas_kv, label='Krūties vėžio duomenys')
    plt.plot(epochos_ir, mokymo_tikslumas_ir, label='Irisų duomenys')
    plt.xlabel('Epochos')
    plt.ylabel('Klasifikavimo tikslumas')
    plt.title('Klasifikavimo tikslumo priklausomybė nuo epochų skaičiaus')
    plt.legend()

    plt.tight_layout()
    plt.show()

    epochos = int(input("Įveskite epochų skaičių: "))  # epochos turi būti sveikasis skaičius

    mokymosi_greiciai = [0.001, 0.01,0.1]

# Saugome rezultatus kiekvienam mokymosi greičiui
    paklaidos_rezultatai = {}
    tikslumo_rezultatai = {}


    for greitis in mokymosi_greiciai:
        print(f"\nTiriame mokymosi greitį: {greitis}")
        np.random.seed(12)
    
    # Mokome neuroną su Krūties vėžio duomenimis
        svoriai_kv, poslinkis_kv, mokymo_klaidos_kv, mokymo_tikslumas_kv = Neurono_mokymas(X_mok_krut_vezio, t_mok_krut_vezio, greitis, epochos)
    
    # Kaupiame paklaidas ir tikslumą
        paklaidos_rezultatai[greitis] = mokymo_klaidos_kv
        tikslumo_rezultatai[greitis] = mokymo_tikslumas_kv

#  paklaidų priklausomybės nuo epochų atvaizdavimas
    for greitis in mokymosi_greiciai:
        epochos = range(1, len(paklaidos_rezultatai[greitis]) + 1)
        plt.plot(epochos, paklaidos_rezultatai[greitis], label=f'Greitis = {greitis}')

    plt.xlabel('Epochos')
    plt.ylabel('Paklaidos reikšmės')
    plt.title('Paklaidos priklausomybė nuo epochų skaičiaus (Krūties vėžio duomenys)')
    plt.legend()
    plt.show()

#  klasifikavimo tikslumo priklausomybės nuo epochų atvaizdavimas
    for greitis in mokymosi_greiciai:
        epochos = range(1, len(tikslumo_rezultatai[greitis]) + 1)
        plt.plot(epochos, tikslumo_rezultatai[greitis], label=f'Greitis = {greitis}')

    plt.xlabel('Epochos')
    plt.ylabel('Tikslumo reikšmės')
    plt.title('Tikslumo priklausomybė nuo epochų skaičiaus (Krūties vėžio duomenys)')
    plt.legend()
    plt.show()

    epochos = int(input("Įveskite epochų skaičių: "))  # epochos turi būti sveikasis skaičius

#irisu
    mokymosi_greiciai_iris = [0.001, 0.01, 0.1]

# Saugome rezultatus kiekvienam mokymosi greičiui
    paklaidos_rezultatai_iris = {}
    tikslumo_rezultatai_iris = {}

    for greitis in mokymosi_greiciai_iris:
        print(f"\nTiriame mokymosi greitį su Irisų duomenimis: {greitis}")
        np.random.seed(12)
    
    # Mokome neuroną su Irisų duomenimis
        svoriai_ir, poslinkis_ir, mokymo_klaidos_ir, mokymo_tikslumas_ir = Neurono_mokymas(X_mok_iris, t_mok_iris, greitis, epochos)
    
    # Kaupiame paklaidas ir tikslumą
        paklaidos_rezultatai_iris[greitis] = mokymo_klaidos_ir
        tikslumo_rezultatai_iris[greitis] = mokymo_tikslumas_ir

# Grafinis paklaidų priklausomybės nuo epochų atvaizdavimas (Irisų duomenys)
    for greitis in mokymosi_greiciai_iris:
        epochos_range = range(1, len(paklaidos_rezultatai_iris[greitis]) + 1)  
        plt.plot(epochos_range, paklaidos_rezultatai_iris[greitis], label=f'Greitis = {greitis}')  

    plt.xlabel('Epochos')
    plt.ylabel('Paklaidos reikšmės')
    plt.title('Paklaidos priklausomybė nuo epochų skaičiaus (Irisų duomenys)')
    plt.legend()
    plt.show()

    
    for greitis in mokymosi_greiciai_iris:
        epochos_range = range(1, len(tikslumo_rezultatai_iris[greitis]) + 1)  
        plt.plot(epochos_range, tikslumo_rezultatai_iris[greitis], label=f'Greitis = {greitis}')  

    plt.xlabel('Epochos')
    plt.ylabel('Tikslumo reikšmės')
    plt.title('Tikslumo priklausomybė nuo epochų skaičiaus (Irisų duomenys)')
    plt.legend()
    plt.show()

  





