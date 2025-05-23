import os
import psycopg2
import psycopg2.extras
import json
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.Error as e:
        logger.error(f"Error connecting to PostgreSQL: {e}")
        raise

def init_db():
    """Initializes the database tables if they don't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id SERIAL PRIMARY KEY,
                    name VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    embedding_id SERIAL PRIMARY KEY,
                    person_id INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
                    embedding JSONB NOT NULL,
                    image_filename VARCHAR(255) NOT NULL,
                    model_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
        logger.info("Database tables checked/created successfully.")
    except psycopg2.Error as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback() # Rollback in case of error
    finally:
        if conn:
            conn.close()

def add_person(name):
    """Adds a new person to the database. Returns person_id or None if error/exists."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO persons (name) VALUES (%s) RETURNING person_id;", (name,))
            person_id = cur.fetchone()[0]
            conn.commit()
            return person_id
    except psycopg2.IntegrityError: # Handles unique constraint violation for name
        logger.warning(f"Person with name '{name}' already exists.")
        conn.rollback()
        # Optionally, fetch the existing person's ID
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT person_id FROM persons WHERE name = %s;", (name,))
            existing_person = cur.fetchone()
            return existing_person['person_id'] if existing_person else None
    except psycopg2.Error as e:
        logger.error(f"Error adding person '{name}': {e}")
        conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def get_person_by_id(person_id):
    """Retrieves a person by their ID."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM persons WHERE person_id = %s;", (person_id,))
            return cur.fetchone()
    except psycopg2.Error as e:
        logger.error(f"Error getting person by ID {person_id}: {e}")
        return None
    finally:
        if conn:
            conn.close()
            
def get_person_by_name(name):
    """Retrieves a person by their name."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT * FROM persons WHERE name = %s;", (name,))
            return cur.fetchone()
    except psycopg2.Error as e:
        logger.error(f"Error getting person by name {name}: {e}")
        return None
    finally:
        if conn:
            conn.close()

def update_person_name(person_id, new_name):
    """Updates the name of a person."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE persons SET name = %s WHERE person_id = %s;", (new_name, person_id))
            conn.commit()
            return True
    except psycopg2.Error as e:
        logger.error(f"Error updating name for person ID {person_id}: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def delete_person(person_id):
    """Deletes a person and their associated embeddings (due to ON DELETE CASCADE)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM persons WHERE person_id = %s;", (person_id,))
            conn.commit()
            return True
    except psycopg2.Error as e:
        logger.error(f"Error deleting person ID {person_id}: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def add_face_embedding(person_id, embedding_vector, image_filename, model_name="Facenet"):
    """Adds a face embedding for a person."""
    conn = get_db_connection()
    try:
        # Convert embedding vector (list of floats) to JSON string for JSONB
        embedding_json = json.dumps(embedding_vector)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO face_embeddings (person_id, embedding, image_filename, model_name) VALUES (%s, %s, %s, %s) RETURNING embedding_id;",
                (person_id, embedding_json, image_filename, model_name)
            )
            embedding_id = cur.fetchone()[0]
            conn.commit()
            return embedding_id
    except psycopg2.Error as e:
        logger.error(f"Error adding face embedding for person ID {person_id}: {e}")
        conn.rollback()
        return None
    finally:
        if conn:
            conn.close()

def get_all_persons_with_embeddings():
    """Retrieves all persons and their associated face embeddings."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Using a LEFT JOIN to include persons even if they have no embeddings yet (though unlikely with current workflow)
            # Or focusing on persons who HAVE embeddings
            cur.execute("""
                SELECT p.person_id, p.name, fe.embedding_id, fe.embedding, fe.image_filename, fe.model_name
                FROM persons p
                JOIN face_embeddings fe ON p.person_id = fe.person_id
                ORDER BY p.name, fe.created_at;
            """)
            # Group results by person
            persons_data = {}
            for row in cur.fetchall():
                pid = row['person_id']
                if pid not in persons_data:
                    persons_data[pid] = {
                        "person_id": pid,
                        "name": row["name"],
                        "embeddings": []
                    }
                persons_data[pid]["embeddings"].append({
                    "embedding_id": row["embedding_id"],
                    "embedding_vector": row["embedding"], # Already a list/dict from JSONB
                    "image_filename": row["image_filename"],
                    "model_name": row["model_name"]
                })
            return list(persons_data.values())
    except psycopg2.Error as e:
        logger.error(f"Error getting all persons with embeddings: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_all_face_embeddings_for_recognition():
    """Retrieves all face embeddings along with person_id and name for recognition."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT fe.embedding_id, fe.person_id, p.name as person_name, fe.embedding, fe.model_name
                FROM face_embeddings fe
                JOIN persons p ON fe.person_id = p.person_id;
            """)
            embeddings_data = []
            for row in cur.fetchall():
                embeddings_data.append({
                    "embedding_id": row["embedding_id"],
                    "person_id": row["person_id"],
                    "person_name": row["person_name"],
                    "embedding_vector": row["embedding"], # JSONB converts to Python list/dict
                    "model_name": row["model_name"]
                })
            return embeddings_data
    except psycopg2.Error as e:
        logger.error(f"Error getting all face embeddings for recognition: {e}")
        return []
    finally:
        if conn:
            conn.close()

def delete_face_embedding(embedding_id):
    """Deletes a specific face embedding by its ID."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM face_embeddings WHERE embedding_id = %s;", (embedding_id,))
            conn.commit()
            # Check if any rows were affected
            return cur.rowcount > 0
    except psycopg2.Error as e:
        logger.error(f"Error deleting face embedding ID {embedding_id}: {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# Call init_db() when this module is loaded to ensure tables exist
# This is a common pattern, but for Flask apps, often done once at app startup.
# For simplicity here, we can call it. Consider moving to app factory if scaling.
if __name__ == '__main__':
    print("Initializing database (if run directly)...")
    init_db()
    print("Database initialization complete.")
    # Example usage:
    # new_person_id = add_person("Lalit Dumka")
    # if new_person_id:
    #     print(f"Added person with ID: {new_person_id}")
    #     # Dummy embedding for testing
    #     dummy_embedding = [0.1] * 128 # Facenet typically 128 or 512
    #     add_face_embedding(new_person_id, dummy_embedding, "lalit_dumka_01.jpg", "Facenet")
    #     print("Added dummy embedding.")

    # print("\nAll persons with embeddings:")
    # for p_data in get_all_persons_with_embeddings():
    #      print(p_data)

    # print("\nEmbeddings for recognition:")
    # for emb_data in get_all_face_embeddings_for_recognition():
    #      print(emb_data)