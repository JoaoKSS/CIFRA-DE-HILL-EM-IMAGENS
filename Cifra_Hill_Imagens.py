import imageio
import numpy as np

# Função usada para determinar se a transformação adicional deve ser aplicada.
Funcao = 1

# Definição da matriz chave fixa 4x4
Mod = 256
A = np.loadtxt('Chave.txt', dtype=int)
n = A.shape[0]  # Dimensão da matriz chave

# Função para calcular a inversa da matriz usando Gauss-Jordan
def gauss_jordan_inverse(matrix, mod):
    n = matrix.shape[0]
    augmented = np.hstack((matrix, np.eye(n)))
    for i in range(n):
        # Normaliza a linha atual
        inv = pow(int(augmented[i, i]), -1, mod)
        augmented[i] = (augmented[i] * inv) % mod
        
        # Zera as outras linhas na mesma coluna
        for j in range(n):
            if i != j:
                factor = augmented[j, i]
                augmented[j] = (augmented[j] - factor * augmented[i]) % mod

    return augmented[:, n:]

# Calcula a inversa da matriz chave
A_inv = gauss_jordan_inverse(A, Mod).astype(int)

np.savetxt('Chave_Inversa_modular.txt', A_inv, fmt='%d')

#---------------Ler Imagem para Criptografar---------------
# Lê a imagem a ser criptografada.
img = imageio.v2.imread('img1.jpg')
# img = imageio.v2.imread('imagem.jpg')

# Obtém as dimensões da imagem.
nl = l = img.shape[0]
w = img.shape[1]

# Ajusta o número de linhas da imagem para ser múltiplo de n.
if l % n:
    nl = (int((l - 1) / n) + 1) * n

# Cria uma nova imagem com o tamanho ajustado, preenchida com zeros.
img2 = np.zeros((nl, w, 3))
img2[:l, :w, :] += img  # Copia a imagem original para a nova imagem.

# Função usada para gerar a transformação adicional.
def f(x, y):
    # Calcula um valor de transformação baseado na matriz chave e nas coordenadas do pixel
    return A[x % 4, y % 4] * (x * x + y * y + 5 * x + 8 * y)

# Aplica a transformação adicional se Funcao for verdadeiro.
if Funcao:
    img3 = np.zeros((nl, w, 3))
    for x in range(nl):
        for y in range(w):
            v = round(f(x, y))  # Arredonda para o inteiro mais próximo.
            img3[x, y, 0] = v
            img3[x, y, 1] = v
            img3[x, y, 2] = v
    img2 = (img2 + img3) % 256  # Aplica a transformação à imagem.
    img2 = img2.astype(np.uint8)  # Converte para o tipo de dados uint8.
    imageio.imwrite('Transformação.png', img2)  # Salva a imagem transformada.

#-------------Criptografando-------------
# Define o tipo de dados para uint8.
Criptografado = np.zeros((nl, w, 3), dtype=np.uint8)
for i in range(int(nl / n)):
    # Criptografa cada canal de cor separadamente usando a cifra de Hill.
    Crip1 = (np.matmul(A % Mod, img2[i * n:(i + 1) * n, :, 0] % Mod)) % Mod
    Crip1 = Crip1.astype(np.uint8)  # Converte para uint8
    
    Crip2 = (np.matmul(A % Mod, img2[i * n:(i + 1) * n, :, 1] % Mod)) % Mod
    Crip2 = Crip2.astype(np.uint8)  # Converte para uint8
    
    Crip3 = (np.matmul(A % Mod, img2[i * n:(i + 1) * n, :, 2] % Mod)) % Mod
    Crip3 = Crip3.astype(np.uint8)  # Converte para uint8
    
    # Redimensiona os arrays criptografados.
    Crip1 = np.resize(Crip1, (Crip1.shape[0], Crip1.shape[1], 1))
    Crip2 = np.resize(Crip2, (Crip2.shape[0], Crip2.shape[1], 1))
    Crip3 = np.resize(Crip3, (Crip3.shape[0], Crip3.shape[1], 1))
    
    # Combina os canais criptografados e adiciona ao resultado final.
    Criptografado[i * n:(i + 1) * n, :] += np.concatenate((Crip1, Crip2, Crip3), axis=2)

# Salva a imagem criptografada.
imageio.imwrite('Criptografado.png', Criptografado)

#-------------Descriptografando-------------
# Lê a imagem criptografada.
Crip = imageio.v2.imread('Criptografado.png')

# Obtém o número de linhas da imagem criptografada.
nl = int(Crip.shape[0])

# Define o tipo de dados para uint8.
Descriptografado = np.zeros((nl, w, 3), dtype=np.uint8)
for i in range(int(nl / n)):
    # Descriptografa cada canal de cor separadamente usando a inversa da cifra de Hill.
    Crip_int = Crip[i * n:(i + 1) * n, :, 0].astype(np.int32)  # Converte para int32
    Desc1 = (np.matmul(A_inv, Crip_int) % Mod).astype(np.uint8)  # Agora tudo é inteiro

    Crip_int = Crip[i * n:(i + 1) * n, :, 1].astype(np.int32)  # Converte para int32
    Desc2 = (np.matmul(A_inv, Crip_int) % Mod).astype(np.uint8)  # Agora tudo é inteiro

    Crip_int = Crip[i * n:(i + 1) * n, :, 2].astype(np.int32)  # Converte para int32
    Desc3 = (np.matmul(A_inv, Crip_int) % Mod).astype(np.uint8)  # Agora tudo é inteiro
    
    # Redimensiona os arrays descriptografados.
    Desc1 = np.resize(Desc1, (Desc1.shape[0], Desc1.shape[1], 1))
    Desc2 = np.resize(Desc2, (Desc2.shape[0], Desc2.shape[1], 1))
    Desc3 = np.resize(Desc3, (Desc3.shape[0], Desc3.shape[1], 1))
    
    # Converte após redimensionar.
    Desc1 = Desc1.astype(np.uint8)
    Desc2 = Desc2.astype(np.uint8)
    Desc3 = Desc3.astype(np.uint8)
    
    # Combina os canais descriptografados e adiciona ao resultado final.
    Descriptografado[i * n:(i + 1) * n, :] += np.concatenate((Desc1, Desc2, Desc3), axis=2)

# Remove a transformação adicional se Funcao for verdadeiro.
if Funcao:
    Descriptografado = (Descriptografado - img3) % 256

# Retorna as dimensões originais para a imagem descriptografada.
Descriptografado = Descriptografado[:l, :w, :]

# Converte Descriptografado para uint8 antes de salvar.
Descriptografado = Descriptografado.astype(np.uint8)
imageio.imwrite('Descriptografado.png', Descriptografado)

print("OPERAÇÕES EFETUADAS COM SUCESSO!!")
