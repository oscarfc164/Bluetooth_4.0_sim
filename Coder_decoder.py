# Librer√≠a utilizada
import binascii

# Objeto encargado de codificar y decodificar mensaje
class Bluetooth_sim:

    def __init__(self, fuente, sumidero):
        # Texto de entrada "tweet" 
        self.fuente = fuente
        # Texto de salida en binario
        self.sumidero = sumidero

    def coder(self):
        #Read from txt
        with open(self.fuente, "r") as file:
            cadena = file.read()
            print("El mensaje es:",cadena)
        #Coding the text read
            self.coder = bin(int(binascii.hexlify(cadena.encode()), 16))
            # La salida tiene "0b" como formato 
            out = self.coder
            print("El mensaje codificado con formato de impresi√≥n binaria: ", out)
            # Mensaje binario limpio
            out_clean = out.replace('b','')
            print("El mensaje codificado es: ", out_clean)

        return self.coder
    
    def decoder(self):
        decode = int(self.coder, 2)
        text = decode.to_bytes((decode.bit_length() + 7) // 8, 'big').decode()

        with open(sumidero, 'w') as file: 
            file.write(text)


fuente = 'tweet_in.txt'
sumidero = 'tweet_out.txt'

#########################################
# Codificaci√≥n de fuente de informaci√≥n #
#########################################

# Secuencia de bits de informaci√≥n en salida de codificador de fuente
bf = Bluetooth_sim(fuente, sumidero)

############################################
# Simulaci√≥n de canal de transmisi√≥n ideal #
############################################

# Se copia informaci√≥n de codificador de salida de fuente
bfp = bf

bfp.coder()
bfp.decoder()


##### Efecto de hacer y deshacer la codificaciÛn de fuente #####

""" En el ejemplo de prueba utilizado, "HOLA" est· compuesto por caracteres 
ASCII que tienen una representaciÛn binaria directa. Debido a lo anterior 
el proceso de codificaciÛn y decodificaciÛn no altera el mensaje original. 
Por lo tanto, es correcto que tanto el archivo tweet_in.txt como el archivo 
tweet_out.txt contengan la cadena "HOLA" en este caso."""




