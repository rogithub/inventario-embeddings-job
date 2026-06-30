import os
import httpx
import psycopg2
import psycopg2.extras

BGE_URL      = os.environ["BGE_EMBEDDINGS_URL"].rstrip("/")
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB   = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASS = os.environ["POSTGRES_PASSWORD"]

FETCH_LIMIT = 500
BATCH_SIZE  = 64


def get_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
    )


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        f"{BGE_URL}/embed-batch",
        json={"texts": texts},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def run() -> None:
    conn = get_conn()
    total = 0

    while True:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT p.Id,
                       p.Nombre
                       || COALESCE(' ' || um.Nombre, '')
                       || COALESCE(' ' || p.Marca, '') AS texto
                FROM Productos p
                LEFT JOIN UnidadesMedida um ON um.Id = p.UnidadMedidaId
                WHERE p.embedding IS NULL
                LIMIT %s
            """, (FETCH_LIMIT,))
            rows = cur.fetchall()

        if not rows:
            break

        for i in range(0, len(rows), BATCH_SIZE):
            batch  = rows[i : i + BATCH_SIZE]
            texts  = [r["texto"] for r in batch]
            ids    = [r["id"] for r in batch]

            vectors = embed_batch(texts)

            with conn.cursor() as cur:
                for product_id, vec in zip(ids, vectors):
                    cur.execute("""
                        UPDATE Productos
                        SET embedding = %s::vector,
                            EmbeddingGeneratedAt = NOW()
                        WHERE Id = %s
                    """, (vector_literal(vec), product_id))
            conn.commit()

            total += len(batch)
            print(f"[ingest] {total} procesados (+{len(batch)})", flush=True)

    print(f"[ingest] Completo. Total: {total}", flush=True)
    conn.close()


if __name__ == "__main__":
    run()
