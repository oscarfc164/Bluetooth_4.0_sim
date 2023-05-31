import random 
import numpy as np
import binascii

def random_sequence(length):
    zeros = length // 2
    ones = length - zeros 
    sequence = [0] * zeros + [1] * ones
    random.shuffle(sequence)
    bf = ''.join(str(bit) for bit in sequence)
    return bf


def source_coder(text_in):
    #Read from txt
    with open(text_in, "r") as file:
        cadena = file.read()
        print("El mensaje es:",cadena)
    #Coding the text read
        coder = bin(int(binascii.hexlify(cadena.encode()), 16))
        # La salida tiene "0b" como formato 
        out = coder
        print("El mensaje codificado con formato de impresión binaria: ", out)
        # Mensaje binario limpio
        out_clean = out.replace('b','')
        print("El mensaje codificado es: ", out_clean)

    return out_clean

def channel_encoder(sequence):
    #Separo bf en vectores de 4 bits para utilizar Hamming 7x4

    sequence_length = len(sequence)//4
    u_vector = []

    for i in range(sequence_length):
        start = i * 4
        end = start + 4
        u = sequence[start:end]
        u_vector.append(u)

    #Matriz G (7x4)

    G = np.array([
    [1, 0, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 1]
    ])

    v_vector = []
    for u in u_vector:
        u_bits = np.array(list(u)).astype(int)
        #print(u_bits)
        v_bits = np.dot(u_bits, G) % 2
        #print(v_bits)
        v = ''.join(str(bit) for bit in v_bits)
        v_vector.append(v)

    bc = ''.join(v_vector) 
    return bc

def ideal_transmition_channel(bc):
    bc_prima = bc
    return bc_prima

def simetric_binary_channel(bc):
    prob = 0.001

    bc_prima = ''
    for bit in bc:
        ran_num = random.random()

        if ran_num < (1-prob):
            e_bit = bit
        else:
            e_bit = 1 - int(bit)

        bc_prima += str(e_bit)

    return bc_prima

def channel_decoder(bc_prima):
    # Matriz H (3x7) - Para detección y corrección de errores
    H = np.array([
        [1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 1, 1]
    ])

    #Calculo de sindrome de paridad 
    bc_prima_length = len(bc_prima)//7
    v_prima_vector = []

    for i in range(bc_prima_length):
        start = i * 7
        end = start + 7
        u = bc_prima[start:end]
        v_prima_vector.append(u)

    v_prima_vector = np.array([[int(bit) for bit in bits] for bits in v_prima_vector])
    
    corrected_v_prima = []
    for v in v_prima_vector:
        a0 = v[0]
        a1 = v[1]
        a2 = v[2]
        a3 = v[3]
        p0 = v[4]
        p1 = v[5]
        p2 = v[6]

        s0 = p0+a0+a1+a2
        s1 = p1+a1+a2+a3
        s2 = p2+a0+a1+a3
        S = [s0, s1, s2]
        #v_t = np.transpose(v)
        #S = np.dot(H,v_t)
        #print(S)
        S = np.remainder(S, 2)
        S = ''.join([str(bit) for bit in S])
        print("Antes correcion: ",v)
        print("sindrome:" ,S)
        print(v[0])

        #Deteccion de error y correccion
        if (S == '001'):
            v[6] = 1 - v[6]
        elif(S == '010'): 
            v[5] = 1 - v[5]
        elif(S == '011'): 
            v[3] = 1 - v[3]
        elif(S == '100'): 
            v[4] = 1 - v[4]
        elif(S == '101'): 
            v[0] = 1 - v[0]
        elif(S == '110'): 
            v[2] = 1 - v[2]
        elif(S == '111'): 
            v[1] = 1 - v[1]

        print("Despues correcion: ",v)

        corrected_v_prima.append(v)

    bfR = ''.join([str(bit) for bits in corrected_v_prima for bit in bits[:4]])
    return bfR

def source_decoder(bfT, text_out):
    decode = int(bfT, 2)
    text = decode.to_bytes((decode.bit_length() + 7) // 8, 'big').decode()

    with open(text_out, 'w') as file: 
        file.write(text)

    with open(text_out, "r") as file:
            cadena_sumidero = file.read()
    
    print("El mensaje luego de el proceso de decodificación de canal y fuente es: ", cadena_sumidero)

def opcion_1():
    print("Ha seleccionado la opción 1")
    
    sequence = random_sequence(32)

    bc = channel_encoder(sequence)
    print(bc)

    bc_prima = simetric_binary_channel(bc)
    print(bc_prima)


    bfT = channel_decoder(bc_prima)
    print("Secuencia enviada: ", sequence)
    print("Secuencia recibida: ", bfT)


    num_bits = len(bc)
    num_bits_diferentes = sum(bit1 != bit2 for bit1, bit2 in zip(bc, bc_prima))
    porcentaje_error = (num_bits_diferentes / num_bits) * 100
    print("% antes de correccion de errores",porcentaje_error)
    num_bits2 = len(sequence)
    num_bits_diferentes2 = sum(bit3 != bit4 for bit3, bit4 in zip(sequence, bfT))
    porcentaje_error2 = (num_bits_diferentes2 / num_bits2) * 100
    print("% despues de correccion de errores",porcentaje_error2)

def opcion_2():
    print("Ha seleccionado la opción 2")
    text_in = 'fuente.txt'
    text_out = 'sumidero.txt'


    sequence = source_coder(text_in)
    bc = channel_encoder(sequence)
    bc_prima = simetric_binary_channel(bc)
    bfT = channel_decoder(bc_prima)

    print("Secuencia enviada: ", sequence)
    print("Secuencia recibida: ", bfT)


    num_bits = len(bc)
    num_bits_diferentes = sum(bit1 != bit2 for bit1, bit2 in zip(bc, bc_prima))
    porcentaje_error = (num_bits_diferentes / num_bits) * 100
    print("% antes de correccion de errores",porcentaje_error)
    num_bits2 = len(sequence)
    num_bits_diferentes2 = sum(bit3 != bit4 for bit3, bit4 in zip(sequence, bfT))
    porcentaje_error2 = (num_bits_diferentes2 / num_bits2) * 100
    print("% despues de correccion de errores",porcentaje_error2)


    source_decoder(bfT, text_out)



def opcion_predeterminada():
    print("Opción inválida")

# Menú de casos
def menu(opcion):
    switch = {
        1: opcion_1,
        2: opcion_2,
    }
    # Obtener la función correspondiente a la opción seleccionada
    funcion = switch.get(opcion, opcion_predeterminada)
    # Ejecutar la función
    funcion()

# Menú
print("Menú de opciones:")
print("1. Opción 1: Codificación de canal")
print("2. Opción 2: Codificación de fuente y canal")

opcion_seleccionada = int(input("Seleccione una opción: "))

menu(opcion_seleccionada)