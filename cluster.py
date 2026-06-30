import os
import numpy as np
import psycopg2
import psycopg2.extras
from sklearn.cluster import KMeans

POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB   = os.environ["POSTGRES_DB"]
POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASS = os.environ["POSTGRES_PASSWORD"]

K      = int(os.environ.get("K_CLUSTERS", "40"))
COMMIT = os.environ.get("COMMIT") == "1"


def get_conn():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
    )


def parse_vector(s: str) -> np.ndarray:
    return np.fromstring(s[1:-1], sep=",", dtype=np.float32)


def run() -> None:
    conn = get_conn()

    print("[cluster] Cargando embeddings...", flush=True)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT Id, Nombre, embedding::text
            FROM Productos
            WHERE embedding IS NOT NULL
        """)
        rows = cur.fetchall()

    print(f"[cluster] {len(rows)} productos cargados.", flush=True)

    ids     = [r["id"] for r in rows]
    nombres = [r["nombre"] for r in rows]
    X       = np.array([parse_vector(r["embedding"]) for r in rows], dtype=np.float32)

    print(f"[cluster] Corriendo k-means con k={K}...", flush=True)
    km     = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    print("[cluster] K-means completo.", flush=True)

    # Nombre tentativo: producto más cercano al centroide de cada cluster
    cluster_info = []
    for k in range(K):
        indices  = np.where(labels == k)[0]
        centroid = km.cluster_centers_[k]
        dists    = np.linalg.norm(X[indices] - centroid, axis=1)
        top_idx  = indices[np.argsort(dists)[:5]]
        top_names = [nombres[i] for i in top_idx]
        cluster_info.append({
            "k":         k,
            "indices":   indices,
            "top_names": top_names,
            "count":     len(indices),
            "nombre":    top_names[0],
        })

    # Reporte
    SEP = "=" * 70
    print(f"\n{SEP}", flush=True)
    print(f"REPORTE DE CLUSTERING  k={K}  ({len(ids)} productos)", flush=True)
    print(SEP, flush=True)
    for info in sorted(cluster_info, key=lambda x: -x["count"]):
        print(f"\nFamilia {info['k']:02d} ({info['count']} productos)", flush=True)
        print(f"  Nombre tentativo : {info['nombre']}", flush=True)
        print(f"  Top 5 cercanos   : {' | '.join(info['top_names'])}", flush=True)
        muestra = [nombres[i] for i in info["indices"][:8]]
        print(f"  Muestra          : {', '.join(muestra)}", flush=True)
    print(f"\n{SEP}", flush=True)

    if not COMMIT:
        print("[cluster] DRY RUN — no se guardaron cambios. Correr con COMMIT=1 para guardar.", flush=True)
        conn.close()
        return

    print("[cluster] Guardando familias en DB...", flush=True)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE FamiliasSemanticas CASCADE")

        familia_ids: dict[int, object] = {}
        for info in cluster_info:
            cur.execute(
                "INSERT INTO FamiliasSemanticas (Nombre) VALUES (%s) RETURNING Id",
                (info["nombre"],)
            )
            familia_ids[info["k"]] = cur.fetchone()[0]

        for i, product_id in enumerate(ids):
            cur.execute(
                "UPDATE Productos SET FamiliaSemanticaId = %s WHERE Id = %s",
                (familia_ids[int(labels[i])], product_id)
            )

    conn.commit()
    print(f"[cluster] Guardado. {K} familias creadas, {len(ids)} productos asignados.", flush=True)
    conn.close()


if __name__ == "__main__":
    run()
