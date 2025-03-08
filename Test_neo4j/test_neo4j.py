from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "testpassword123"))

with driver.session() as session:
    result = session.run("RETURN 1")
    print(result.single()[0])

driver.close()