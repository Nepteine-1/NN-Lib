FROM ubuntu:24.04

RUN apt update && apt upgrade -y && apt install -y cmake g++
WORKDIR /data/

# Création d'un utilisateur non-root
RUN useradd -u 1111 nn-lib

# Changement de permissions pour l'utilisateur non-root dans /data/ pour accéder aux scripts de compilation
RUN chown -R nn-lib:nn-lib /data/
USER nn-lib

# Copie des fichiers nécessaires dans le répertoire du container "/data/"
COPY . .

CMD ["sh", "docker_build.sh"]
