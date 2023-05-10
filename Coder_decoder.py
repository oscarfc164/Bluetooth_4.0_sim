import binascii

class Bluetooth_sim:

    def __init__(self, text_in, text_out):
        self.text_in = text_in
        self.text_out = text_out

    def coder(self):
        #Read from txt
        with open(self.text_in, "r") as file:
            cadena = file.read()
        
            #print(cadena)
        #Coding the text read
            self.coder = bin(int(binascii.hexlify(cadena.encode()), 16))
            print(self.coder)

        return self.coder
    
    def decoder(self):
        decode = int(self.coder, 2)
        text = decode.to_bytes((decode.bit_length() + 7) // 8, 'big').decode()

        with open(text_out, 'w') as file: 
            file.write(text)


text_in = 'tweet_in.txt'
text_out = 'tweet_out.txt'

sim = Bluetooth_sim(text_in, text_out)

sim.coder()
sim.decoder()




