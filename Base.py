from pathlib import Path
import pandas as pd
from pyreaddbc import dbc2dbf
from dbfread import DBF

pasta_raw = Path("dados/raw")
pasta_interim = Path("dados/interim")

pasta_interim.mkdir(parents=True, exist_ok=True)

for arquivo_dbc in pasta_raw.glob("*.dbc"):
    print(f"Convertendo {arquivo_dbc.name}...")

    arquivo_dbf = pasta_interim / f"{arquivo_dbc.stem}.dbf"
    arquivo_csv = pasta_interim / f"{arquivo_dbc.stem}.csv"

    dbc2dbf(str(arquivo_dbc), str(arquivo_dbf))

    tabela = DBF(
        str(arquivo_dbf),
        encoding="latin1",
        char_decode_errors="ignore"
    )

    df_temp = pd.DataFrame(iter(tabela))

    df_temp.to_csv(
        arquivo_csv,
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    print(f"Salvo: {arquivo_csv.name}")

    arquivos_csv = list(Path("dados/interim").glob("RD*.csv"))

dfs = []

for arquivo in arquivos_csv:
    df_temp = pd.read_csv(arquivo, sep=";", encoding="utf-8-sig")
    df_temp["arquivo_origem"] = arquivo.name
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)

df.shape

pasta_processed = Path("dados/processed")
pasta_processed.mkdir(parents=True, exist_ok=True)

caminho_saida = pasta_processed / "sih_sus_mt_internacoes_completo.csv"

df.to_csv(
    caminho_saida,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"Arquivo salvo em: {caminho_saida}")