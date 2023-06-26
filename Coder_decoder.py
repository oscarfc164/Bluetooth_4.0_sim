# Librería utilizada
import binascii

# Objeto encargado de codificar y decodificar mensaje
class Bluetooth_sim:

    def __init__(self, text_in, text_out):
        # Texto de entrada "tweet" 
        self.text_in = text_in
        # Texto de salida en binario
        self.text_out = text_out

    def source_coder(self):
        #Read from txt
        with open(self.text_in, "r") as file:
            cadena = file.read()
            print("El mensaje es:",cadena)
        #Coding the text read
            self.coder = bin(int(binascii.hexlify(cadena.encode()), 16))
            # La salida tiene "0b" como formato 
            out = self.coder
            print("El mensaje codificado con formato de impresión binaria: ", out)
            # Mensaje binario limpio
            out_clean = out.replace('b','')
            print("El mensaje codificado es: ", out_clean)

        return self.coder

    
    def source_decoder(self):
        decode = int(self.coder, 2)
        print(decode)
        text = decode.to_bytes((decode.bit_length() + 7) // 8, 'big').decode()

        with open(text_out, 'w') as file: 
            file.write(text)


text_in = 'fuente.txt'
text_out = 'sumidero.txt'

#########################################
# Codificación de fuente de información #
#########################################

# Secuencia de bits de información en salida de codificador de fuente
bf = Bluetooth_sim(text_in, text_out)

############################################
# Simulación de canal de transmisión ideal #
############################################

# Se copia información de codificador de salida de fuente
bfp = bf

bfp.source_coder()
bfp.source_decoder()


##### Efecto de hacer y deshacer la codificaci�n de fuente #####

""" En el ejemplo de prueba utilizado, "HOLA" est� compuesto por caracteres 
ASCII que tienen una representaci�n binaria directa. Debido a lo anterior 
el proceso de codificaci�n y decodificaci�n no altera el mensaje original. 
Por lo tanto, es correcto que tanto el archivo tweet_in.txt como el archivo 
tweet_out.txt contengan la cadena "HOLA" en este caso."""




