# Section 1: Backend Engineering -- Detailed Interview Answers

**Candidate: Deepaksakthi | AI Engineer | 1.5 years at SuperOps (Kotlin, Spring Boot, jOOQ, Apache Pulsar)**

---

## Question 1: What is JPA? Explain its internal working -- persistence context, entity lifecycle, dirty checking, and flush modes.

**JPA (Java Persistence API)** is a specification -- not a library -- that defines a standard way for Java/Kotlin applications to manage relational data using object-relational mapping (ORM). It lives under the `jakarta.persistence` (formerly `javax.persistence`) package and provides annotations like `@Entity`, `@Table`, `@Id`, and interfaces like `EntityManager`.

### Persistence Context

The persistence context is the first-level cache and the core working area of JPA. It is a set of managed entity instances where, for any given persistent identity, there is exactly one entity instance. The `EntityManager` is the interface to this context. When you call `em.find(User.class, 1L)`, JPA checks the persistence context first before hitting the database.

```kotlin
@PersistenceContext
lateinit var em: EntityManager

fun example() {
    val user1 = em.find(User::class.java, 1L) // SQL fires
    val user2 = em.find(User::class.java, 1L) // No SQL -- served from persistence context
    assert(user1 === user2) // Same reference -- identity guarantee
}
```

### Entity Lifecycle

An entity moves through four states:

1. **New/Transient** -- Just created with `new`, not associated with any persistence context.
2. **Managed/Persistent** -- Attached to a persistence context. Changes are tracked. Reached via `persist()`, `find()`, or `merge()`.
3. **Detached** -- Was managed, but the persistence context was closed or the entity was evicted. Changes are NOT tracked.
4. **Removed** -- Scheduled for deletion. Still managed until the transaction commits.

```
Transient --persist()--> Managed --remove()--> Removed
                            |                      |
                         detach()              commit()
                            |                      |
                            v                      v
                        Detached               Deleted from DB
```

### Dirty Checking

When a transaction commits (or a flush occurs), JPA compares the current state of every managed entity against a snapshot taken when the entity was first loaded. If any field differs, an UPDATE SQL statement is generated automatically. This means you never need to call an explicit `update()` -- just modify the managed entity.

Internally, Hibernate (the most common JPA implementation) keeps a copy of the original hydrated state (`Object[]`) for each entity. At flush time, it walks through all managed entities, field by field, and compares current values to the snapshot.

### Flush Modes

- **FlushModeType.AUTO** (default): The persistence provider flushes before any query execution to ensure consistency, and at transaction commit.
- **FlushModeType.COMMIT**: Flush only happens at transaction commit. Queries may return stale data, but this offers better performance for batch operations.

```kotlin
em.flushMode = FlushModeType.COMMIT // Only flush on commit
```

At SuperOps, we used jOOQ instead of JPA, but understanding JPA was essential when evaluating our tech stack choices and when working with upstream libraries that assumed JPA semantics.

---

## Question 2: What is JPQL? How does it differ from native SQL and Criteria API?

**JPQL (Java Persistence Query Language)** is an object-oriented query language defined by the JPA specification. It operates on entities and their fields rather than database tables and columns.

### JPQL Example

```java
// JPQL -- references entity class name and field names
String jpql = "SELECT u FROM User u WHERE u.email = :email AND u.isActive = true";
List<User> users = em.createQuery(jpql, User.class)
    .setParameter("email", "deepak@superops.ai")
    .getResultList();
```

### Native SQL

Native SQL queries operate directly on database tables and columns. You write raw SQL that gets passed straight to the database.

```java
// Native SQL -- references table name and column names
String sql = "SELECT * FROM users u WHERE u.email = ? AND u.is_active = true";
List<User> users = em.createNativeQuery(sql, User.class)
    .setParameter(1, "deepak@superops.ai")
    .getResultList();
```

### Criteria API

The Criteria API provides a programmatic, type-safe way to build queries using Java/Kotlin objects instead of string-based queries.

```java
CriteriaBuilder cb = em.getCriteriaBuilder();
CriteriaQuery<User> cq = cb.createQuery(User.class);
Root<User> root = cq.from(User.class);
cq.select(root).where(
    cb.and(
        cb.equal(root.get("email"), "deepak@superops.ai"),
        cb.isTrue(root.get("isActive"))
    )
);
List<User> users = em.createQuery(cq).getResultList();
```

### Key Differences

| Aspect | JPQL | Native SQL | Criteria API |
|--------|------|-----------|--------------|
| Operates on | Entities/fields | Tables/columns | Entities (programmatic) |
| Type safety | No (strings) | No (strings) | Partial (metamodel helps) |
| Database portability | Yes | No | Yes |
| Complex queries | Moderate | Full SQL power | Verbose but flexible |
| Dynamic queries | Hard (string concat) | Hard (string concat) | Designed for this |
| Readability | Good | Good | Poor for complex queries |

JPQL is essentially "SQL for objects." It is database-portable because the JPA provider translates it to the appropriate dialect. However, it falls short for database-specific features like window functions, CTEs, or recursive queries -- which is where native SQL or a tool like jOOQ shines.

The Criteria API's verbosity is its biggest drawback. A 3-line JPQL query can become 15 lines of Criteria code. This was one reason why, at SuperOps, we preferred jOOQ -- it gives you type-safe, composable queries without the verbosity of the Criteria API, and with the full expressive power of SQL.

---

## Question 3: How does Hibernate work internally? Explain the session, first-level cache, second-level cache, and the proxy pattern for lazy loading.

### Hibernate Session

The `Session` is Hibernate's core interface (it implements JPA's `EntityManager`). It wraps a JDBC connection and acts as a unit of work. Internally, when you open a session, Hibernate creates a `PersistenceContext` (also called `StatefulPersistenceContext`) that tracks all loaded entities.

```kotlin
val sessionFactory: SessionFactory = // built from configuration
val session = sessionFactory.openSession()
val tx = session.beginTransaction()

val user = session.get(User::class.java, 1L) // Loads entity into Session
user.name = "Deepak"  // Dirty checking will detect this
tx.commit()  // Flushes changes -- UPDATE SQL generated
session.close()
```

### First-Level Cache (Session Cache)

Every Session has a built-in first-level cache. It is a `Map<EntityKey, Object>` internally. When you load an entity, it is stored in this map. Any subsequent `get()` or `find()` for the same ID returns the cached instance without hitting the database.

- Scoped to the session (transaction-level typically)
- Cannot be disabled
- Prevents duplicate entity references within a session (identity map pattern)
- Cleared when session closes

```kotlin
val user1 = session.get(User::class.java, 1L) // SELECT fires
val user2 = session.get(User::class.java, 1L) // No SELECT -- first-level cache hit
println(user1 === user2) // true -- same object reference
```

### Second-Level Cache (L2 Cache)

The second-level cache is shared across sessions within the same `SessionFactory`. It is optional and must be explicitly configured. Popular implementations include Ehcache, Infinispan, and Hazelcast.

```kotlin
@Entity
@Cacheable
@Cache(usage = CacheConcurrencyStrategy.READ_WRITE)
class User(
    @Id val id: Long,
    val name: String
)
```

The flow is: Session (L1) -> L2 Cache -> Database. L2 cache stores dehydrated data (not entity objects) as `Object[]` arrays keyed by entity ID. This means it does not hold references to session-specific objects and is safe for cross-session sharing.

### Proxy Pattern for Lazy Loading

When you declare a `@ManyToOne` or `@OneToMany` relationship as `FetchType.LAZY`, Hibernate does not load the associated entity immediately. Instead, it creates a **proxy** -- a dynamically generated subclass (using Byte Buddy or javassist) that extends the entity class.

```kotlin
@Entity
class Ticket(
    @Id val id: Long,

    @ManyToOne(fetch = FetchType.LAZY)
    val assignee: User  // Not loaded immediately -- proxy created
)

val ticket = session.get(Ticket::class.java, 1L)
// ticket.assignee is a Hibernate proxy, not a real User
println(ticket.assignee.javaClass.name) // Something like "User$HibernateProxy$abc123"

ticket.assignee.name // NOW the actual SELECT for User fires (proxy is initialized)
```

The proxy intercepts method calls via an `LazyInitializer` handler. When you access any non-ID field, it triggers initialization -- executing the actual SQL query. If the session is closed when you access the proxy, you get the dreaded `LazyInitializationException`.

---

## Question 4: JPA vs Hibernate -- JPA is a specification, Hibernate is an implementation. But what does that actually mean in practice? Can you use JPA without Hibernate?

### What "Specification vs Implementation" Means in Practice

JPA is like a Java interface -- it defines the contract (annotations, interfaces, behavior rules) but provides zero executable code. Hibernate is like the class that implements that interface.

Concretely:
- **JPA** defines `@Entity`, `@Table`, `@Id`, `EntityManager`, `TypedQuery`, etc. These live in `jakarta.persistence.*`.
- **Hibernate** provides the actual engine that reads those annotations, generates SQL, manages connections, implements caching, etc. Its classes live in `org.hibernate.*`.

When you write:
```kotlin
@Entity
@Table(name = "users")
class User(
    @Id @GeneratedValue val id: Long,
    val name: String
)
```

This is pure JPA. But when your application starts, it is Hibernate's `SessionFactory` that scans this class, builds the metadata, creates the SQL schema mapping, and handles all persistence operations behind the scenes.

### Can You Use JPA Without Hibernate?

Absolutely yes. JPA is designed to be provider-agnostic. You can swap Hibernate for:

- **EclipseLink** (reference implementation, used by GlassFish)
- **OpenJPA** (Apache's implementation)
- **DataNucleus** (used by Google App Engine)

Your `persistence.xml` specifies which provider to use:

```xml
<!-- Using Hibernate -->
<persistence-unit name="myPU">
    <provider>org.hibernate.jpa.HibernatePersistenceProvider</provider>
</persistence-unit>

<!-- Using EclipseLink -->
<persistence-unit name="myPU">
    <provider>org.eclipse.persistence.jpa.PersistenceProvider</provider>
</persistence-unit>
```

If you stick strictly to JPA annotations and the `EntityManager` API, switching providers requires only a dependency swap and configuration change.

### Where It Gets Blurry

In practice, most teams end up using Hibernate-specific features:

- `@Cache` for second-level caching
- `@Where` for soft deletes
- `@Formula` for computed columns
- `Session` API directly for batch operations
- Hibernate-specific `@Type` annotations

Once you use these, you are coupled to Hibernate. This is the practical trade-off: JPA gives you portability in theory, but Hibernate gives you power features you often need.

### My Perspective from SuperOps

At SuperOps, we sidestepped this entire JPA vs Hibernate debate by choosing jOOQ. We wanted explicit SQL control, and the JPA abstraction -- while powerful -- sometimes made it harder to understand what SQL was actually hitting the database. When you are debugging a complex ticket-triaging query with multiple joins and aggregations, having a 1:1 mapping between your code and the generated SQL is invaluable.

---

## Question 5: Why Hibernate? What are its alternatives (EclipseLink, OpenJPA, MyBatis, jOOQ)?

### Why Teams Choose Hibernate

Hibernate is the dominant ORM in the Java ecosystem for good reasons:

1. **Mature and battle-tested** -- 20+ years of development, enormous community
2. **Automatic schema-to-object mapping** -- Write entities, get CRUD for free
3. **Dirty checking and change tracking** -- Modify objects, SQL is generated
4. **Caching** -- Built-in L1 cache, pluggable L2 cache
5. **Database portability** -- Switch databases by changing a dialect
6. **Spring integration** -- Spring Data JPA makes Hibernate nearly invisible

### Alternatives Compared

**EclipseLink**
- JPA reference implementation (what the spec was written against)
- Strong in Java EE/Jakarta EE environments
- Better out-of-the-box support for multi-tenancy and advanced mapping
- Smaller community than Hibernate, fewer blog posts and Stack Overflow answers

**OpenJPA**
- Apache's JPA implementation
- Was popular in the WebSphere ecosystem
- Largely dormant in terms of active development
- Not recommended for new projects

**MyBatis**
- Not an ORM -- it is a SQL mapper
- You write SQL in XML files or annotations, MyBatis maps results to objects
- Full SQL control, no magic, no lazy loading surprises
- Popular in Chinese tech companies and banking software

```xml
<!-- MyBatis XML mapper -->
<select id="findUserByEmail" resultType="User">
    SELECT id, name, email FROM users WHERE email = #{email}
</select>
```

**jOOQ (Java Object Oriented Querying)**
- Code-generates type-safe Java/Kotlin classes from your database schema
- You write SQL in Java/Kotlin with full IDE support and compile-time checking
- No ORM abstraction -- you think in SQL, but it is type-safe
- Excellent for complex queries, reporting, analytics

```kotlin
// jOOQ -- looks like SQL, but it is Kotlin code with full type safety
val result = dsl.select(USERS.NAME, TICKETS.COUNT())
    .from(USERS)
    .join(TICKETS).on(TICKETS.ASSIGNEE_ID.eq(USERS.ID))
    .where(USERS.IS_ACTIVE.isTrue)
    .groupBy(USERS.NAME)
    .fetch()
```

### When to Choose What

| Use Case | Best Fit |
|----------|----------|
| Simple CRUD, rapid prototyping | Hibernate + Spring Data JPA |
| Complex queries, analytics, reporting | jOOQ |
| Legacy DB, need raw SQL | MyBatis |
| Jakarta EE environment | EclipseLink |
| Microservice with simple data needs | Spring Data JPA |
| Need explicit SQL control + type safety | jOOQ |

At SuperOps, we chose jOOQ because our domain (IT ops management) involved complex queries -- ticket triaging with dynamic filters, alert correlation, reporting dashboards -- where Hibernate's abstraction was more hindrance than help.

---

## Question 6: Hibernate vs jOOQ -- What is the fundamental philosophical difference? When would you choose one over the other?

### The Core Philosophical Difference

**Hibernate says:** "Think in objects, forget about SQL. I will figure out the SQL for you."

**jOOQ says:** "Think in SQL, but I will make your SQL type-safe and composable in your programming language."

This is not just a minor difference -- it represents two fundamentally different views of how application code should relate to a relational database.

Hibernate follows the **Active Record / Data Mapper pattern**. You define entities that mirror your domain model, and Hibernate manages the gap between objects and tables (the "object-relational impedance mismatch"). You work with objects, and trust Hibernate to generate efficient SQL.

jOOQ follows the **database-first paradigm**. The database schema is the source of truth. jOOQ generates code from your schema, and you write SQL using that generated code. There is no entity lifecycle, no dirty checking, no persistence context -- you get records in, records out.

### Code Comparison

```kotlin
// Hibernate: You think in objects
@Transactional
fun reassignTickets(fromUserId: Long, toUserId: Long) {
    val tickets = ticketRepository.findByAssigneeId(fromUserId)
    val newAssignee = userRepository.findById(toUserId).get()
    tickets.forEach { it.assignee = newAssignee } // Dirty checking handles UPDATE
}
// Problem: This fires N+1 queries if relationships are not eagerly fetched,
// loads all tickets into memory, and generates N UPDATE statements

// jOOQ: You think in SQL
fun reassignTickets(fromUserId: Long, toUserId: Long) {
    dsl.update(TICKETS)
        .set(TICKETS.ASSIGNEE_ID, toUserId)
        .where(TICKETS.ASSIGNEE_ID.eq(fromUserId))
        .execute()
}
// Single UPDATE statement, exactly what you would write in SQL
```

### When to Choose Hibernate

- CRUD-heavy applications with simple domain models
- Rapid prototyping -- Spring Data JPA gets you running in minutes
- When you want database portability
- When your team thinks in terms of domain objects, not SQL
- When you want automatic change tracking and cascading

### When to Choose jOOQ

- Complex queries: multi-join, window functions, CTEs, recursive queries
- Performance-critical paths where you need to control exact SQL
- Reporting/analytics features
- When your database schema is the source of truth (database-first design)
- When you want compile-time SQL verification
- When debugging -- jOOQ queries map 1:1 to the generated SQL

### My SuperOps Experience

At SuperOps, the decision was clear. Our ticket triaging system needed dynamic filter composition, complex aggregations for dashboards, and precise control over query performance. With Hibernate, we would have fought against the abstraction. With jOOQ, we wrote exactly the SQL we wanted, with the safety net of type checking. When a column was renamed in a migration, jOOQ's regenerated code broke at compile time -- not at runtime in production.

---

## Question 7: What is the N+1 query problem in Hibernate? How do you detect and fix it?

### What Is the N+1 Problem?

The N+1 query problem occurs when Hibernate executes 1 query to load a list of N parent entities, and then N additional queries to load a relationship for each parent -- totaling N+1 queries instead of the optimal 1 or 2.

```kotlin
@Entity
class Ticket(
    @Id val id: Long,
    val title: String,

    @ManyToOne(fetch = FetchType.LAZY)
    val assignee: User
)

// The problem:
val tickets = em.createQuery("SELECT t FROM Ticket t", Ticket::class.java).resultList
// Query 1: SELECT * FROM tickets (returns 100 tickets)

tickets.forEach { println(it.assignee.name) }
// Query 2: SELECT * FROM users WHERE id = 1
// Query 3: SELECT * FROM users WHERE id = 2
// ... 100 more queries!
// Total: 101 queries for what should be a single JOIN
```

### How to Detect It

**1. Enable SQL logging:**
```yaml
# application.yml
spring:
  jpa:
    show-sql: true
    properties:
      hibernate:
        format_sql: true
        generate_statistics: true
```

**2. Use a query counter in tests:**
```kotlin
@Test
fun `should not have N+1 problem when loading tickets`() {
    val queryCount = QueryCountHolder.getGrandTotal()
    val tickets = ticketService.getAllWithAssignees()
    tickets.forEach { it.assignee.name } // Force initialization
    val totalQueries = QueryCountHolder.getGrandTotal() - queryCount
    assertThat(totalQueries).isLessThanOrEqualTo(2)
}
```

**3. Use tools like** `datasource-proxy`, `p6spy`, or Hibernate's built-in statistics to count queries per request.

### How to Fix It

**Fix 1: JOIN FETCH in JPQL**
```java
SELECT t FROM Ticket t JOIN FETCH t.assignee
// Single query with a JOIN -- loads both tickets and users
```

**Fix 2: @EntityGraph**
```kotlin
@EntityGraph(attributePaths = ["assignee"])
fun findAll(): List<Ticket>
```

**Fix 3: @BatchSize (Hibernate-specific)**
```kotlin
@ManyToOne(fetch = FetchType.LAZY)
@BatchSize(size = 25)
val assignee: User
// Instead of N queries, Hibernate loads assignees in batches:
// SELECT * FROM users WHERE id IN (1,2,3,...,25)
// SELECT * FROM users WHERE id IN (26,27,...,50)
```

**Fix 4: Subselect fetching**
```kotlin
@Fetch(FetchMode.SUBSELECT)
val assignee: User
// Loads all assignees in a single subselect query
```

### Why This Is a jOOQ Advantage

With jOOQ, the N+1 problem simply does not exist because you write explicit SQL. You decide whether to JOIN or not. There is no hidden lazy loading. At SuperOps, our queries always did exactly what we wrote -- no surprises in production when a seemingly simple endpoint suddenly fires hundreds of queries.

---

## Question 8: Explain Hibernate's dirty checking mechanism. How does it know which fields changed?

### How Dirty Checking Works Internally

When Hibernate loads an entity from the database, it performs **hydration** -- converting the JDBC `ResultSet` into an `Object[]` array representing the entity's state. This snapshot is stored in the persistence context alongside the entity itself.

The internal data structure looks conceptually like this:

```
PersistenceContext:
  EntityEntry(User#1):
    entity: User@7a81e3 (the actual managed object)
    loadedState: Object[] { "Deepak", "deepak@superops.ai", true }  // snapshot
    status: MANAGED
```

### The Comparison Process

At flush time (before commit or before a query in AUTO flush mode), Hibernate's `DefaultFlushEntityEventListener` iterates over all managed entities and performs a **field-by-field comparison**:

```java
// Simplified Hibernate internal logic
for (entity in persistenceContext.managedEntities()) {
    Object[] currentState = entityPersister.getPropertyValues(entity)
    Object[] loadedState = entityEntry.getLoadedState()

    for (i in 0 until properties.size) {
        if (!propertyTypes[i].isEqual(currentState[i], loadedState[i])) {
            // Field i is dirty -- schedule UPDATE
            dirtyFields.add(i)
        }
    }
}
```

### Key Details

**1. Snapshot comparison, not change-event tracking:** Hibernate does NOT use setter interception or bytecode-enhanced setters by default. It compares the full state at flush time. This means even if you set a field to the same value (`user.name = user.name`), Hibernate still compares it and correctly determines nothing changed.

**2. Default behavior generates full UPDATE statements:**
```sql
-- Even if only 'name' changed, Hibernate generates:
UPDATE users SET name=?, email=?, is_active=? WHERE id=?
-- All columns are included
```

**3. @DynamicUpdate for selective updates:**
```kotlin
@Entity
@DynamicUpdate
class User(...)
// Now Hibernate generates: UPDATE users SET name=? WHERE id=?
// Only changed columns are included
```

**4. Bytecode enhancement (optional optimization):**
Hibernate can be configured to use bytecode enhancement to intercept field writes, eliminating the need for snapshot comparison. This is more memory-efficient and faster for entities with many fields.

```xml
<!-- Maven plugin for bytecode enhancement -->
<plugin>
    <groupId>org.hibernate.orm.tooling</groupId>
    <artifactId>hibernate-enhance-maven-plugin</artifactId>
    <configuration>
        <enableDirtyTracking>true</enableDirtyTracking>
    </configuration>
</plugin>
```

With enhancement, Hibernate adds a `$$_hibernate_tracker` field to the entity that records which fields were modified, making flush-time comparison O(1) instead of O(n).

### Performance Implications

Dirty checking every managed entity at flush time has a cost. If your session contains 10,000 managed entities, Hibernate must snapshot-compare all of them. This is why:
- Detach entities you are done with: `em.detach(entity)`
- Use read-only transactions when you only need to read: `@Transactional(readOnly = true)`
- Consider stateless sessions for bulk operations: `sessionFactory.openStatelessSession()`

At SuperOps with jOOQ, dirty checking was not relevant -- we explicitly wrote UPDATE statements for exactly the fields we wanted to change.

---

## Question 9: What are Hibernate's fetching strategies? Explain EAGER vs LAZY loading and their pitfalls.

### EAGER Loading

With `FetchType.EAGER`, the associated entity/collection is loaded immediately along with the parent entity, always, regardless of whether you need it.

```kotlin
@Entity
class Ticket(
    @Id val id: Long,

    @ManyToOne(fetch = FetchType.EAGER) // Always loaded with Ticket
    val assignee: User,

    @OneToMany(fetch = FetchType.EAGER) // Always loaded with Ticket
    val comments: List<Comment>
)
```

### LAZY Loading

With `FetchType.LAZY`, the associated entity is not loaded until it is actually accessed. Hibernate creates a proxy (for `@ManyToOne`) or a wrapper collection (for `@OneToMany`).

```kotlin
@Entity
class Ticket(
    @Id val id: Long,

    @ManyToOne(fetch = FetchType.LAZY) // Loaded only when accessed
    val assignee: User,

    @OneToMany(fetch = FetchType.LAZY) // Loaded only when accessed
    val comments: List<Comment>
)
```

### JPA Default Fetch Types

- `@ManyToOne` and `@OneToOne`: **EAGER** by default (this surprises many developers)
- `@OneToMany` and `@ManyToMany`: **LAZY** by default

### Pitfalls of EAGER Loading

**1. Cannot be turned off per query.** Once you declare EAGER, it is always eager. You cannot say "load this Ticket without the assignee this time."

**2. Cartesian product explosion.** Multiple EAGER collections cause a cartesian product:
```kotlin
@OneToMany(fetch = FetchType.EAGER)
val comments: List<Comment> // 10 comments

@OneToMany(fetch = FetchType.EAGER)
val attachments: List<Attachment> // 5 attachments

// Result set: 10 x 5 = 50 rows for a single Ticket!
```

**3. Kills performance silently.** Loading a list of 100 tickets eagerly loads all their associated entities -- you may pull thousands of rows from the database for a "simple" list endpoint.

### Pitfalls of LAZY Loading

**1. LazyInitializationException:**
```kotlin
fun getTicket(id: Long): Ticket {
    val ticket = ticketRepository.findById(id).get()
    return ticket
} // Session closes here

// In the controller/serializer:
ticket.assignee.name // BOOM! LazyInitializationException
// The session is closed, the proxy cannot initialize
```

**2. N+1 problem** (covered in Question 7).

**3. Open Session in View (OSIV) anti-pattern:** Spring Boot enables `spring.jpa.open-in-view=true` by default, which keeps the session open during view rendering. This masks LazyInitializationException but leads to unpredictable query execution during JSON serialization.

### Best Practices

```kotlin
// 1. Default everything to LAZY
@ManyToOne(fetch = FetchType.LAZY)
val assignee: User

// 2. Use JOIN FETCH or @EntityGraph when you need eager loading
@Query("SELECT t FROM Ticket t JOIN FETCH t.assignee WHERE t.id = :id")
fun findTicketWithAssignee(@Param("id") id: Long): Ticket

// 3. Use DTOs/projections to load exactly what you need
@Query("SELECT new TicketSummaryDTO(t.id, t.title, u.name) " +
       "FROM Ticket t JOIN t.assignee u")
fun findTicketSummaries(): List<TicketSummaryDTO>
```

### Why This Matters for jOOQ Users

At SuperOps, we did not deal with EAGER vs LAZY because jOOQ does not have an entity graph or lazy loading. You explicitly select the columns and joins you need. This might sound tedious, but it means every query is predictable and optimized for its use case. No hidden SQL, no proxy surprises.

---

## Question 10: What is the difference between merge(), persist(), save(), and update() in Hibernate?

### persist() -- JPA Standard

```kotlin
val user = User(name = "Deepak") // Transient state
em.persist(user) // Becomes Managed
// INSERT is scheduled (may not fire immediately -- waits for flush)
// user.id is now populated (if using IDENTITY generation)
```

- Defined in JPA specification (`EntityManager.persist()`)
- Only works for **new/transient** entities
- If entity already has an ID and exists in DB, throws `EntityExistsException`
- Does NOT return anything (void)
- The passed object itself becomes managed

### save() -- Hibernate-Specific (Deprecated)

```kotlin
val user = User(name = "Deepak")
val generatedId: Serializable = session.save(user) // Returns the generated ID
```

- Defined in Hibernate's `Session` interface, NOT in JPA
- Returns the generated identifier (`Serializable`)
- Deprecated since Hibernate 6.0 -- use `persist()` instead
- Immediately executes INSERT if the ID generation strategy requires it (e.g., IDENTITY)

### merge() -- JPA Standard

```kotlin
val detachedUser = getDetachedUserFromSomewhere() // Has id=1, not managed
detachedUser.name = "Updated Name"

val managedUser = em.merge(detachedUser)
// managedUser is now managed, detachedUser is STILL detached
// They are different object references!
```

- Works for both **new** and **detached** entities
- Returns a **new managed copy** -- the original object remains detached
- If the entity exists in DB, loads it and copies the state (SELECT + UPDATE)
- If the entity does not exist, creates a new one (INSERT)
- Common mistake: continuing to use the original detached object instead of the returned managed copy

### update() -- Hibernate-Specific (Deprecated)

```kotlin
val detachedUser = getDetachedUserFromSomewhere()
detachedUser.name = "Updated"
session.update(detachedUser) // Reattaches the SAME object to the session
```

- Defined in Hibernate's `Session` interface, NOT in JPA
- Reattaches the passed object directly (unlike `merge` which creates a copy)
- Throws `NonUniqueObjectException` if the session already contains an entity with the same ID
- Deprecated since Hibernate 6.0

### Summary Table

| Method | Standard | Returns | Input State | After Call |
|--------|----------|---------|-------------|------------|
| `persist()` | JPA | void | Transient | Same object is Managed |
| `save()` | Hibernate | ID | Transient | Same object is Managed |
| `merge()` | JPA | Managed copy | Detached/Transient | New managed copy returned |
| `update()` | Hibernate | void | Detached | Same object is Managed |

### Practical Advice

```kotlin
// Modern code should use only:
em.persist(newEntity)           // For new entities
em.merge(detachedEntity)        // For detached entities
em.remove(managedEntity)        // For deletion
// Let dirty checking handle updates to managed entities
```

At SuperOps, since we used jOOQ, our insert/update operations were explicit SQL-like DSL calls -- `dsl.insertInto(USERS).set(...).execute()` and `dsl.update(USERS).set(...).where(...).execute()` -- which made the semantics unambiguous.

---

## Question 11: How does transaction management work in Spring Boot? Explain @Transactional, propagation levels, and isolation levels.

### How @Transactional Works

Spring uses **AOP (Aspect-Oriented Programming)** proxies to manage transactions. When you annotate a method with `@Transactional`, Spring wraps it in a proxy that:

1. Opens a transaction (or joins an existing one)
2. Executes your method
3. Commits on success, rollbacks on RuntimeException

```kotlin
@Service
class TicketService(private val ticketRepo: TicketRepository) {

    @Transactional
    fun createTicket(dto: CreateTicketDTO): Ticket {
        val ticket = Ticket(title = dto.title, status = "OPEN")
        ticketRepo.save(ticket)
        // If this method completes normally -> COMMIT
        // If RuntimeException thrown -> ROLLBACK
        return ticket
    }
}
```

**Critical gotcha:** `@Transactional` does NOT work for self-invocation (calling a `@Transactional` method from within the same class) because the proxy is bypassed:

```kotlin
@Service
class TicketService {
    fun processTickets() {
        createTicket(dto) // Direct call -- NO TRANSACTION! Proxy bypassed.
    }

    @Transactional
    fun createTicket(dto: CreateTicketDTO) { ... }
}
```

### Propagation Levels

Propagation defines what happens when a transactional method calls another transactional method.

```kotlin
@Transactional(propagation = Propagation.REQUIRED) // Default
fun methodA() {
    methodB() // What happens here depends on B's propagation
}
```

| Propagation | Behavior |
|------------|----------|
| **REQUIRED** (default) | Join existing TX, or create new if none exists |
| **REQUIRES_NEW** | Suspend current TX, create a brand new one |
| **NESTED** | Create a savepoint within the current TX |
| **SUPPORTS** | Use TX if one exists, run non-transactional if not |
| **NOT_SUPPORTED** | Suspend current TX, run non-transactional |
| **MANDATORY** | Must run inside existing TX, throw exception if none |
| **NEVER** | Must NOT run inside TX, throw exception if one exists |

**Real-world example from SuperOps:** Audit logging should persist even if the main operation fails:

```kotlin
@Transactional
fun updateTicketStatus(ticketId: Long, newStatus: String) {
    ticketDao.updateStatus(ticketId, newStatus)
    auditService.log(ticketId, "Status changed to $newStatus") // Should persist even if later code fails
}

@Service
class AuditService {
    @Transactional(propagation = Propagation.REQUIRES_NEW) // Independent TX
    fun log(entityId: Long, message: String) {
        auditDao.insert(entityId, message)
    }
}
```

### Isolation Levels

Isolation levels control how concurrent transactions see each other's changes.

| Isolation Level | Dirty Read | Non-Repeatable Read | Phantom Read |
|----------------|-----------|-------------------|-------------|
| READ_UNCOMMITTED | Yes | Yes | Yes |
| READ_COMMITTED | No | Yes | Yes |
| REPEATABLE_READ | No | No | Yes |
| SERIALIZABLE | No | No | No |

```kotlin
@Transactional(isolation = Isolation.REPEATABLE_READ)
fun generateReport(): Report {
    // Ensures consistent reads throughout this transaction
    val tickets = ticketDao.findAll()
    val stats = ticketDao.getStatistics()
    // Both queries see the same snapshot of data
    return Report(tickets, stats)
}
```

Most databases default to **READ_COMMITTED** (PostgreSQL, Oracle) or **REPEATABLE_READ** (MySQL InnoDB). At SuperOps, we used PostgreSQL with its default READ_COMMITTED and relied on explicit locking (`SELECT ... FOR UPDATE` via jOOQ) for critical sections.

---

## Question 12: What is jOOQ's approach to database access? How does its code generation work?

### jOOQ's Philosophy

jOOQ (Java Object Oriented Querying) takes a **database-first** approach. Instead of mapping objects to tables, jOOQ generates Java/Kotlin classes directly from your database schema. You then write type-safe SQL using these generated classes.

The core principle: **SQL is a first-class citizen.** Instead of hiding SQL behind an abstraction, jOOQ embraces it and makes it type-safe.

```kotlin
// jOOQ query -- reads like SQL, but is Kotlin code
val overdueTickets = dsl
    .select(TICKETS.ID, TICKETS.TITLE, USERS.NAME)
    .from(TICKETS)
    .join(USERS).on(TICKETS.ASSIGNEE_ID.eq(USERS.ID))
    .where(TICKETS.DUE_DATE.lt(LocalDate.now()))
    .and(TICKETS.STATUS.ne("CLOSED"))
    .orderBy(TICKETS.PRIORITY.desc())
    .fetch()
```

### How Code Generation Works

jOOQ's code generator connects to your database (or reads DDL files/Flyway migrations) and generates Java/Kotlin classes representing:

**1. Table references:**
```kotlin
// Generated class
object TICKETS : TableImpl<TicketsRecord>("tickets") {
    val ID: TableField<TicketsRecord, Long> = createField("id", BIGINT)
    val TITLE: TableField<TicketsRecord, String> = createField("title", VARCHAR)
    val ASSIGNEE_ID: TableField<TicketsRecord, Long> = createField("assignee_id", BIGINT)
    val STATUS: TableField<TicketsRecord, String> = createField("status", VARCHAR)
    val PRIORITY: TableField<TicketsRecord, Int> = createField("priority", INTEGER)
    val DUE_DATE: TableField<TicketsRecord, LocalDate> = createField("due_date", DATE)
}
```

**2. Record classes** -- type-safe row representations
**3. POJO classes** (optional) -- plain data objects
**4. DAO classes** (optional) -- basic CRUD operations

### Configuration (Gradle Example)

```kotlin
// build.gradle.kts
jooq {
    configurations {
        create("main") {
            jooqConfiguration.apply {
                jdbc.apply {
                    driver = "org.postgresql.Driver"
                    url = "jdbc:postgresql://localhost:5432/superops"
                    user = "dev"
                    password = "dev"
                }
                generator.apply {
                    database.apply {
                        inputSchema = "public"
                        includes = ".*"
                        excludes = "flyway_schema_history"
                    }
                    target.apply {
                        packageName = "com.superops.generated"
                        directory = "src/main/generated"
                    }
                }
            }
        }
    }
}
```

### The Code Generation Pipeline

```
Database Schema
    |
    v
jOOQ Code Generator (reads metadata via JDBC or DDL parsing)
    |
    v
Generated Kotlin/Java files:
  - Tables.kt (table references with typed fields)
  - Records.kt (row representations)
  - Keys.kt (primary/foreign key references)
  - Indexes.kt (index metadata)
  - Routines.kt (stored procedures/functions)
    |
    v
Your application code uses these generated types
    |
    v
jOOQ renders SQL for your target database dialect
```

### Compile-Time Safety

The killer feature: if someone renames a column in a migration, the next code generation run produces different classes. Any code referencing the old column name fails at **compile time**, not at runtime in production.

```kotlin
// If TICKETS.ASSIGNEE_ID is renamed to TICKETS.OWNER_ID in a migration,
// this code fails to compile after regeneration:
dsl.select().from(TICKETS).where(TICKETS.ASSIGNEE_ID.eq(1L))
//                                      ^^^^^^^^^^^ Unresolved reference
```

At SuperOps, this saved us multiple times during schema evolution.

---

## Question 13: Why did SuperOps choose jOOQ over Hibernate/JPA? What are the tradeoffs?

### Why jOOQ Was Chosen

SuperOps is an IT operations management platform. The domain involves complex data access patterns that do not map well to a simple ORM:

**1. Complex queries are the norm, not the exception.**
Ticket triaging requires dynamic filtering across multiple dimensions -- priority, status, assignee, team, SLA breach status, custom fields. Building these queries with JPA Criteria API would have been extremely verbose. With JPQL, dynamic composition means string concatenation. With jOOQ, it is natural:

```kotlin
fun findTickets(filter: TicketFilter): List<TicketDTO> {
    var query = dsl.select(TICKETS.ID, TICKETS.TITLE, USERS.NAME)
        .from(TICKETS)
        .join(USERS).on(TICKETS.ASSIGNEE_ID.eq(USERS.ID))

    // Dynamic composition -- add conditions only if filter values are present
    val conditions = mutableListOf<Condition>()
    filter.status?.let { conditions.add(TICKETS.STATUS.eq(it)) }
    filter.priority?.let { conditions.add(TICKETS.PRIORITY.ge(it)) }
    filter.assigneeId?.let { conditions.add(TICKETS.ASSIGNEE_ID.eq(it)) }
    filter.slaBreached?.let { conditions.add(TICKETS.SLA_DUE.lt(DSL.currentTimestamp())) }

    return query.where(conditions).fetchInto(TicketDTO::class.java)
}
```

**2. Alerting and event correlation require advanced SQL.**
Window functions, CTEs, lateral joins -- features that Hibernate/JPQL cannot express without falling back to native queries.

```kotlin
// Find tickets where alert count spiked -- uses window function
val spikeDetection = dsl
    .select(
        ALERTS.TICKET_ID,
        ALERTS.CREATED_AT,
        DSL.count().over().partitionBy(ALERTS.TICKET_ID)
            .orderBy(ALERTS.CREATED_AT)
            .rowsBetweenUnboundedPreceding().andCurrentRow()
            .`as`("running_count")
    )
    .from(ALERTS)
    .fetch()
```

**3. Kotlin alignment.** jOOQ's DSL pairs beautifully with Kotlin's concise syntax, extension functions, and null safety. Hibernate's annotation-heavy, Java-centric model felt foreign in Kotlin.

**4. Predictability.** No lazy loading surprises, no N+1 problems, no wondering "what SQL is Hibernate actually generating?"

### The Tradeoffs

| Aspect | jOOQ Cost | Hibernate Benefit |
|--------|-----------|-------------------|
| Boilerplate for simple CRUD | More code needed | Spring Data JPA gives CRUD free |
| Caching | No built-in entity cache | L1 + L2 caching |
| Change tracking | Manual UPDATE statements | Automatic dirty checking |
| Schema changes | Must regenerate code | Entities are your schema |
| Learning curve | Must know SQL well | Can avoid SQL initially |
| Database portability | Tied to dialect (mitigated by jOOQ renderer) | Mostly portable |

**The honest answer:** For 70% of web applications doing basic CRUD, Spring Data JPA with Hibernate is the faster choice. SuperOps fell in the 30% where SQL complexity justified jOOQ. The extra boilerplate for simple operations was a worthwhile price for the control and safety we gained on complex operations.

---

## Question 14: Explain the Repository pattern in Spring Data JPA. How does Spring auto-generate implementations from interface method names?

### The Repository Pattern

The Repository pattern abstracts data access behind a collection-like interface, decoupling domain logic from persistence details.

```kotlin
interface TicketRepository : JpaRepository<Ticket, Long> {
    fun findByStatus(status: String): List<Ticket>
    fun findByAssigneeIdAndStatusIn(assigneeId: Long, statuses: List<String>): List<Ticket>
    fun countByPriorityGreaterThanEqual(priority: Int): Long
}
```

You declare the interface -- Spring provides the implementation at runtime. There is no class to write.

### How Spring Auto-Generates Implementations

**Step 1: Classpath scanning.** At startup, `@EnableJpaRepositories` (auto-configured in Spring Boot) triggers `RepositoryConfigurationDelegate` to scan for interfaces extending `Repository` or its subtypes.

**Step 2: Proxy creation.** For each repository interface, Spring creates a JDK dynamic proxy (or a CGLIB proxy) backed by `SimpleJpaRepository` as the default implementation for standard CRUD methods (`save`, `findById`, `delete`, etc.).

**Step 3: Query derivation.** For custom method names like `findByStatus`, Spring's `PartTree` query derivation engine parses the method name:

```
findByAssigneeIdAndStatusIn
  |        |         |    |
  find   property  AND  property + IN operator
         (assigneeId)    (status)
```

The parser splits on keywords: `find...By`, `And`, `Or`, `OrderBy`, `Between`, `LessThan`, `GreaterThan`, `In`, `Like`, `IsNull`, `IsTrue`, etc.

**Step 4: JPQL generation.** The parsed tree is converted to JPQL:
```sql
SELECT t FROM Ticket t WHERE t.assignee.id = ?1 AND t.status IN ?2
```

**Step 5: Method interception.** When you call the method, the proxy intercepts it, looks up the pre-built query, binds parameters by position, and executes it via `EntityManager`.

### Under the Hood -- Key Classes

```
RepositoryFactorySupport
  └── JpaRepositoryFactory
       └── Creates proxy with:
           - SimpleJpaRepository (CRUD base)
           - QueryExecutorMethodInterceptor (custom query methods)
               └── PartTreeJpaQuery (derived from method name)
               └── or SimpleJpaQuery (from @Query annotation)
```

### Customization Options

```kotlin
interface TicketRepository : JpaRepository<Ticket, Long> {

    // 1. Derived query -- parsed from method name
    fun findByStatusAndPriorityGreaterThan(status: String, priority: Int): List<Ticket>

    // 2. Explicit JPQL
    @Query("SELECT t FROM Ticket t WHERE t.slaDeadline < CURRENT_TIMESTAMP AND t.status <> 'CLOSED'")
    fun findSlaBreached(): List<Ticket>

    // 3. Native SQL
    @Query(value = "SELECT * FROM tickets WHERE tsv @@ to_tsquery(:query)", nativeQuery = true)
    fun fullTextSearch(@Param("query") query: String): List<Ticket>

    // 4. Modifying queries
    @Modifying
    @Query("UPDATE Ticket t SET t.status = :status WHERE t.id IN :ids")
    fun bulkUpdateStatus(@Param("ids") ids: List<Long>, @Param("status") status: String): Int
}
```

### Relation to SuperOps

At SuperOps, we used jOOQ instead of Spring Data JPA repositories. However, we still followed the Repository pattern conceptually -- wrapping our jOOQ queries in DAO/Repository classes to maintain separation of concerns. The difference was that our repositories contained explicit jOOQ queries rather than relying on method-name magic.

---

## Question 15: What is connection pooling? How does HikariCP work in Spring Boot?

### Why Connection Pooling?

Creating a database connection is expensive. It involves TCP handshake, SSL negotiation, authentication, and protocol setup -- typically 20-50ms per connection. In a web application handling hundreds of requests per second, creating and closing connections for each request would be catastrophic.

A **connection pool** maintains a set of pre-created, reusable database connections. When your code needs a connection, it borrows one from the pool. When done, it returns it rather than closing it.

```
Without pooling:
Request -> Create Connection (50ms) -> Execute Query (5ms) -> Close Connection
Total: 55ms per request

With pooling:
Request -> Borrow Connection (0.1ms) -> Execute Query (5ms) -> Return Connection
Total: 5.1ms per request
```

### HikariCP

HikariCP is the default connection pool in Spring Boot (since 2.0). It is the fastest JDBC connection pool, known for its minimal overhead, zero-allocation steady state, and small codebase (~130KB).

### How HikariCP Works Internally

**1. Pool initialization:**
```yaml
spring:
  datasource:
    hikari:
      minimum-idle: 5          # Minimum idle connections in pool
      maximum-pool-size: 10     # Maximum total connections
      idle-timeout: 300000      # 5 min -- idle connections beyond minimum-idle are closed
      max-lifetime: 1800000     # 30 min -- connections are recycled
      connection-timeout: 30000 # 30s -- max wait for a connection from pool
```

**2. Connection borrowing:** HikariCP uses a custom lock-free data structure called `ConcurrentBag`. When a thread requests a connection:
- First tries **thread-local** (returns the connection this thread last used -- great for cache locality)
- Then scans the **shared list** using a copy-on-write `ArrayList`
- If none available, waits up to `connectionTimeout` for one to be returned

**3. Connection validation:** Before returning a connection to the borrower, HikariCP validates it using `Connection.isValid(timeout)` (JDBC4 validation) or a test query. This prevents your application from using a stale/broken connection.

**4. Connection lifecycle management:**
```
New Connection -> Active (in use by thread) -> Idle (in pool) -> Evicted (max-lifetime reached)
                     ^                            |
                     |                            |
                     +---- returned to pool ------+
```

### Key Configuration for Production

```kotlin
@Configuration
class DataSourceConfig {
    @Bean
    fun hikariConfig(): HikariConfig {
        return HikariConfig().apply {
            jdbcUrl = "jdbc:postgresql://db-host:5432/superops"
            username = "app_user"
            password = System.getenv("DB_PASSWORD")
            maximumPoolSize = 10    // Rule of thumb: connections = (core_count * 2) + disk_spindles
            minimumIdle = 5
            idleTimeout = 300_000
            maxLifetime = 1_740_000 // Slightly less than DB's wait_timeout
            connectionTimeout = 30_000
            leakDetectionThreshold = 60_000 // Log warning if connection not returned in 60s
            poolName = "SuperOps-HikariPool"
        }
    }
}
```

### Pool Sizing

The most common mistake is setting `maximumPoolSize` too high. A PostgreSQL best practice from the official wiki: `connections = (core_count * 2) + effective_spindle_count`. For a 4-core server with SSD, that is about 9-10 connections. More connections mean more context switching and lock contention at the database level.

At SuperOps, we monitored pool metrics via Micrometer + Prometheus -- active connections, idle connections, pending threads, and connection wait time. An alert on `hikaricp.connections.pending` helped us detect pool exhaustion before it impacted users.

---

## Question 16: You used Kotlin with Spring Boot -- what advantages does Kotlin offer over Java for backend development? Any pain points with Spring?

### Advantages of Kotlin

**1. Null safety -- the killer feature:**
```kotlin
// Kotlin forces you to handle nulls at compile time
fun findUser(id: Long): User? { ... } // The ? means it might be null

val user = findUser(42)
// user.name  // Compile error! Must handle null
user?.name    // Safe call -- returns null if user is null
user!!.name   // Explicit assertion -- throws NPE if null (you chose this)
```

This eliminated an entire category of production bugs. In Java, any reference can be null, and you discover it at runtime.

**2. Data classes replace Java boilerplate:**
```kotlin
// Kotlin
data class TicketDTO(
    val id: Long,
    val title: String,
    val status: String,
    val priority: Int
)
// Automatically generates: equals(), hashCode(), toString(), copy(), componentN()

// Java equivalent: 50+ lines (or Lombok dependency)
```

**3. Extension functions for cleaner APIs:**
```kotlin
fun DSLContext.findActiveTickets(): List<TicketRecord> =
    this.selectFrom(TICKETS)
        .where(TICKETS.STATUS.ne("CLOSED"))
        .fetch()

// Usage: dsl.findActiveTickets()
```

**4. Coroutines for async programming:**
```kotlin
suspend fun enrichTicketWithExternalData(ticketId: Long): EnrichedTicket {
    val ticket = ticketDao.findById(ticketId)
    // These run concurrently
    val (customer, slaPolicy) = coroutineScope {
        val customerDeferred = async { customerService.find(ticket.customerId) }
        val slaDeferred = async { slaService.getPolicy(ticket.slaPolicyId) }
        customerDeferred.await() to slaDeferred.await()
    }
    return EnrichedTicket(ticket, customer, slaPolicy)
}
```

**5. Scope functions (`let`, `apply`, `also`, `run`, `with`):**
```kotlin
val config = HikariConfig().apply {
    jdbcUrl = "jdbc:postgresql://localhost/superops"
    maximumPoolSize = 10
    username = "app"
}
// Much cleaner than Java's setter chains
```

**6. Sealed classes for exhaustive when expressions:**
```kotlin
sealed class TicketEvent {
    data class Created(val ticket: Ticket) : TicketEvent()
    data class StatusChanged(val ticketId: Long, val newStatus: String) : TicketEvent()
    data class Assigned(val ticketId: Long, val assigneeId: Long) : TicketEvent()
}

fun handleEvent(event: TicketEvent) = when (event) {
    is TicketEvent.Created -> notifyTeam(event.ticket)
    is TicketEvent.StatusChanged -> updateDashboard(event.ticketId)
    is TicketEvent.Assigned -> notifyAssignee(event.assigneeId)
    // Compiler ensures all cases are handled -- no forgotten branch
}
```

### Pain Points with Spring

**1. All-open plugin requirement.** Spring uses CGLIB proxies that require classes to be open (non-final). Kotlin classes are final by default. Solution: `kotlin-spring` compiler plugin auto-opens `@Component`, `@Service`, `@Configuration`, etc.

**2. Constructor injection verbosity:**
```kotlin
// This fails -- Spring cannot instantiate data class with val
@Entity
data class User(@Id val id: Long, val name: String)
// Needs: no-arg constructor, mutable fields, open class
// Solution: kotlin-jpa plugin generates synthetic no-arg constructors
```

**3. Nullable vs non-null in Spring APIs.** Some Spring methods return nullable types that Kotlin's platform type system (`Type!`) does not catch. You might get an unexpected null from `repository.findById()` that the compiler did not warn about.

**4. Annotation processing.** `kapt` (Kotlin annotation processing) is slower than Java's `apt`. This affects build times, especially with jOOQ code generation and MapStruct.

Despite these pain points, Kotlin was a net positive at SuperOps. The null safety alone prevented dozens of potential production NPEs.

---

## Question 17: You worked with Apache Pulsar at SuperOps. How does Pulsar differ from Kafka? Why was Pulsar chosen?

### Architecture Difference -- The Fundamental Split

The most important architectural difference is that **Pulsar separates compute from storage**.

**Kafka:** Brokers handle both serving (compute) and storing (storage) messages. Data is stored on local broker disks. If a broker dies, you must replicate data to recover.

**Pulsar:** Brokers handle only serving. Storage is delegated to **Apache BookKeeper** -- a distributed log storage system. Brokers are stateless.

```
Kafka Architecture:
Producer -> Broker (serves + stores) -> Consumer
            [local disk with replicas]

Pulsar Architecture:
Producer -> Broker (serves only, stateless) -> Consumer
               |
               v
         BookKeeper (stores data)
         [distributed ledger storage]
```

### Key Differences

| Feature | Kafka | Pulsar |
|---------|-------|--------|
| Storage | Broker-local disk | Apache BookKeeper (separate) |
| Broker state | Stateful | Stateless |
| Scaling | Rebalance partitions (heavy) | Add brokers instantly (lightweight) |
| Multi-tenancy | Limited (topics are flat) | Native (tenant/namespace/topic hierarchy) |
| Subscription models | Consumer groups only | Exclusive, Shared, Failover, Key_Shared |
| Message acknowledgment | Offset-based (sequential) | Individual message ack |
| Geo-replication | MirrorMaker (external tool) | Built-in cross-datacenter replication |
| Tiered storage | Requires plugins | Native support |
| Message queue semantics | No (log only) | Yes (supports both queue and streaming) |

### Why SuperOps Chose Pulsar

**1. Multi-tenancy.** SuperOps is a multi-tenant SaaS platform. Pulsar's native tenant/namespace hierarchy let us isolate customer data at the messaging layer:
```
persistent://superops/customer-A/ticket-events
persistent://superops/customer-B/ticket-events
persistent://superops/shared/system-alerts
```

**2. Flexible subscription models.** For alert processing, we needed shared subscriptions where multiple consumer instances process alerts in parallel. For audit logging, we needed exclusive subscriptions for ordered processing. Pulsar supports both natively.

```kotlin
// Shared subscription -- alerts processed by any available consumer
val consumer = pulsarClient.newConsumer(Schema.JSON(AlertEvent::class.java))
    .topic("persistent://superops/alerts/raw-events")
    .subscriptionName("alert-processor")
    .subscriptionType(SubscriptionType.Shared)
    .subscribe()

// Process alerts
while (true) {
    val msg = consumer.receive()
    try {
        processAlert(msg.value)
        consumer.acknowledge(msg) // Individual ack
    } catch (e: Exception) {
        consumer.negativeAcknowledge(msg) // Retry later
    }
}
```

**3. Individual message acknowledgment.** Unlike Kafka where you commit an offset (acknowledging all messages up to that point), Pulsar lets you ack individual messages. If message 5 fails but messages 6-10 succeed, you only retry message 5.

**4. Easier operations.** Stateless brokers mean scaling up is as simple as adding instances behind a load balancer. No partition rebalancing storms like in Kafka.

**5. Built-in schema registry.** Pulsar has a native schema registry with enforcement, eliminating the need for a separate Confluent Schema Registry.

The tradeoff: Kafka has a much larger ecosystem and community. Pulsar is catching up but has fewer third-party integrations and learning resources.

---

## Question 18: Explain how you designed your REST APIs at SuperOps. What was your approach to error handling, versioning, and authentication?

### API Design Principles

At SuperOps, we followed resource-oriented REST design. The APIs served both the frontend SPA and external integrations (MSPs -- Managed Service Providers -- integrating with their tools).

```
GET    /api/v1/tickets                    # List tickets (with filtering/pagination)
GET    /api/v1/tickets/{id}               # Get single ticket
POST   /api/v1/tickets                    # Create ticket
PUT    /api/v1/tickets/{id}               # Full update
PATCH  /api/v1/tickets/{id}               # Partial update
DELETE /api/v1/tickets/{id}               # Delete ticket
POST   /api/v1/tickets/{id}/assign        # Action (not a resource -- uses verb)
GET    /api/v1/tickets/{id}/comments      # Sub-resource
```

### Error Handling

We used a consistent error response format across all endpoints, implemented via Spring's `@ControllerAdvice`:

```kotlin
data class ApiError(
    val status: Int,
    val code: String,          // Machine-readable: "TICKET_NOT_FOUND"
    val message: String,       // Human-readable: "Ticket with ID 42 not found"
    val timestamp: Instant,
    val traceId: String,       // For log correlation
    val details: List<FieldError>? = null  // Validation errors
)

@RestControllerAdvice
class GlobalExceptionHandler {

    @ExceptionHandler(TicketNotFoundException::class)
    fun handleNotFound(ex: TicketNotFoundException, request: HttpServletRequest): ResponseEntity<ApiError> {
        return ResponseEntity.status(404).body(
            ApiError(
                status = 404,
                code = "TICKET_NOT_FOUND",
                message = ex.message ?: "Resource not found",
                timestamp = Instant.now(),
                traceId = MDC.get("traceId") ?: "unknown"
            )
        )
    }

    @ExceptionHandler(MethodArgumentNotValidException::class)
    fun handleValidation(ex: MethodArgumentNotValidException): ResponseEntity<ApiError> {
        val fieldErrors = ex.bindingResult.fieldErrors.map {
            FieldError(field = it.field, message = it.defaultMessage ?: "Invalid value")
        }
        return ResponseEntity.status(400).body(
            ApiError(
                status = 400,
                code = "VALIDATION_ERROR",
                message = "Request validation failed",
                timestamp = Instant.now(),
                traceId = MDC.get("traceId") ?: "unknown",
                details = fieldErrors
            )
        )
    }
}
```

### Versioning

We used URL path versioning (`/api/v1/`, `/api/v2/`) because it is the most explicit and easiest for API consumers to understand. Header-based versioning (Accept header) was considered but rejected because it is less discoverable and harder to test (you cannot just change the URL in a browser or curl).

```kotlin
@RestController
@RequestMapping("/api/v1/tickets")
class TicketControllerV1 { ... }

@RestController
@RequestMapping("/api/v2/tickets")
class TicketControllerV2 { ... }
```

For minor, backward-compatible changes, we extended existing endpoints rather than creating new versions.

### Authentication

SuperOps used **JWT-based authentication** with Spring Security:

```kotlin
@Configuration
@EnableWebSecurity
class SecurityConfig {

    @Bean
    fun securityFilterChain(http: HttpSecurity): SecurityFilterChain {
        return http
            .csrf { it.disable() } // Stateless API -- CSRF not applicable
            .sessionManagement { it.sessionCreationPolicy(SessionCreationPolicy.STATELESS) }
            .authorizeHttpRequests {
                it.requestMatchers("/api/v1/auth/**").permitAll()
                it.requestMatchers("/api/v1/admin/**").hasRole("ADMIN")
                it.anyRequest().authenticated()
            }
            .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter::class.java)
            .build()
    }
}
```

For external API consumers (MSPs), we issued API keys that were validated against our database and mapped to tenant contexts, ensuring multi-tenant isolation.

---

## Question 19: You used jOOQ at SuperOps -- walk me through how you wrote a complex query (e.g., for ticket triaging or alerting). How did jOOQ's type safety help?

### Real-World Example: Ticket Triage Dashboard Query

The ticket triage view needed to display tickets with dynamic filtering, sorting, pagination, assignee information, SLA status, and alert counts -- all in a single optimized query.

```kotlin
fun findTriageTickets(filter: TriageFilter, page: Int, size: Int): TriageResult {

    // Step 1: Build dynamic conditions
    val conditions = mutableListOf<Condition>()

    filter.teamId?.let {
        conditions.add(TICKETS.TEAM_ID.eq(it))
    }
    filter.statuses?.let {
        conditions.add(TICKETS.STATUS.`in`(it))
    }
    filter.priorities?.let {
        conditions.add(TICKETS.PRIORITY.`in`(it))
    }
    filter.searchText?.let {
        conditions.add(
            TICKETS.TITLE.containsIgnoreCase(it)
                .or(TICKETS.DESCRIPTION.containsIgnoreCase(it))
        )
    }
    filter.slaBreached?.let { breached ->
        if (breached) {
            conditions.add(TICKETS.SLA_DUE_AT.lt(DSL.currentTimestamp()))
        } else {
            conditions.add(TICKETS.SLA_DUE_AT.ge(DSL.currentTimestamp()))
        }
    }

    // Step 2: Alert count subquery
    val alertCount = DSL.select(DSL.count())
        .from(ALERTS)
        .where(ALERTS.TICKET_ID.eq(TICKETS.ID))
        .asField<Int>("alert_count")

    // Step 3: Main query with JOIN, subquery, pagination
    val query = dsl
        .select(
            TICKETS.ID,
            TICKETS.TITLE,
            TICKETS.STATUS,
            TICKETS.PRIORITY,
            TICKETS.CREATED_AT,
            TICKETS.SLA_DUE_AT,
            USERS.ID.`as`("assignee_id"),
            USERS.NAME.`as`("assignee_name"),
            TEAMS.NAME.`as`("team_name"),
            alertCount
        )
        .from(TICKETS)
        .leftJoin(USERS).on(TICKETS.ASSIGNEE_ID.eq(USERS.ID))
        .leftJoin(TEAMS).on(TICKETS.TEAM_ID.eq(TEAMS.ID))
        .where(conditions)
        .orderBy(
            when (filter.sortBy) {
                "priority" -> TICKETS.PRIORITY.desc()
                "created" -> TICKETS.CREATED_AT.desc()
                "sla" -> TICKETS.SLA_DUE_AT.asc()
                else -> TICKETS.CREATED_AT.desc()
            }
        )
        .limit(size)
        .offset(page * size)

    // Step 4: Count query for pagination metadata
    val totalCount = dsl
        .selectCount()
        .from(TICKETS)
        .where(conditions)
        .fetchOne(0, Long::class.java) ?: 0L

    // Step 5: Execute and map
    val tickets = query.fetch().map { record ->
        TriageTicketDTO(
            id = record[TICKETS.ID],
            title = record[TICKETS.TITLE],
            status = record[TICKETS.STATUS],
            priority = record[TICKETS.PRIORITY],
            createdAt = record[TICKETS.CREATED_AT],
            slaDueAt = record[TICKETS.SLA_DUE_AT],
            assigneeName = record["assignee_name", String::class.java],
            teamName = record["team_name", String::class.java],
            alertCount = record["alert_count", Int::class.java] ?: 0
        )
    }

    return TriageResult(tickets = tickets, total = totalCount, page = page, size = size)
}
```

### How Type Safety Helped

**1. Column reference safety.** When we renamed `TICKETS.ASSIGNED_TO` to `TICKETS.ASSIGNEE_ID` during a refactor, the code generator produced new classes, and every query referencing the old name failed at compile time. Without jOOQ, this would have been a runtime error discovered (hopefully) in testing.

**2. Type mismatch prevention.**
```kotlin
// This will NOT compile -- TICKETS.PRIORITY is Int, cannot compare with String
TICKETS.PRIORITY.eq("HIGH")  // Compile error!
TICKETS.PRIORITY.eq(3)       // Correct
```

**3. SQL syntax safety.** jOOQ's fluent API makes it impossible to write syntactically invalid SQL. You cannot forget a `FROM` clause or put a `WHERE` after `ORDER BY`.

**4. Dialect-aware rendering.** The same jOOQ code generates correct SQL whether targeting PostgreSQL, MySQL, or H2 (for tests). Date functions, string operations, and pagination syntax are automatically adapted.

### Alerting Query Example

```kotlin
// Find correlated alerts using a CTE (Common Table Expression)
fun findCorrelatedAlerts(windowMinutes: Int): List<CorrelatedAlertGroup> {
    val alertWindow = DSL.name("alert_window").`as`(
        DSL.select(
            ALERTS.asterisk(),
            DSL.count().over()
                .partitionBy(ALERTS.SOURCE, ALERTS.ALERT_TYPE)
                .orderBy(ALERTS.CREATED_AT)
                .rangeBetween(
                    -windowMinutes * 60, // seconds before
                    0                     // current row
                ).`as`("window_count")
        ).from(ALERTS)
         .where(ALERTS.CREATED_AT.gt(DSL.currentTimestamp().minus(DSL.interval(1, DatePart.HOUR))))
    )

    return dsl.with(alertWindow)
        .select()
        .from(alertWindow)
        .where(DSL.field("window_count", Int::class.java).gt(5))
        .fetchInto(CorrelatedAlertGroup::class.java)
}
```

This kind of query would require a native SQL string in Hibernate -- losing all type safety.

---

## Question 20: How did you handle database migrations in production at SuperOps?

### Migration Tool: Flyway

At SuperOps, we used **Flyway** for database migrations. Flyway follows a simple model: versioned SQL scripts that run in order, tracked in a `flyway_schema_history` table.

```
src/main/resources/db/migration/
├── V1__create_users_table.sql
├── V2__create_tickets_table.sql
├── V3__add_priority_to_tickets.sql
├── V4__create_alerts_table.sql
├── V5__add_sla_fields.sql
├── V6__create_teams_table.sql
└── V7__add_index_on_tickets_status.sql
```

### Migration Script Example

```sql
-- V3__add_priority_to_tickets.sql
ALTER TABLE tickets ADD COLUMN priority INTEGER NOT NULL DEFAULT 3;
ALTER TABLE tickets ADD COLUMN sla_due_at TIMESTAMP;

CREATE INDEX idx_tickets_priority ON tickets(priority);
CREATE INDEX idx_tickets_sla_due ON tickets(sla_due_at) WHERE sla_due_at IS NOT NULL;

-- Backfill existing tickets based on type
UPDATE tickets SET priority = 1 WHERE type = 'INCIDENT';
UPDATE tickets SET priority = 3 WHERE type = 'SERVICE_REQUEST';
```

### Spring Boot Integration

```yaml
# application.yml
spring:
  flyway:
    enabled: true
    locations: classpath:db/migration
    baseline-on-migrate: true
    validate-on-migrate: true
```

Flyway runs automatically at application startup, before Spring Data or jOOQ initializes. This ensures the schema is up-to-date before any application code tries to access the database.

### Integration with jOOQ Code Generation

This was a critical part of our workflow. After writing a migration, we needed to regenerate jOOQ classes:

```
1. Write migration SQL:       V8__add_resolved_at_to_tickets.sql
2. Apply migration to dev DB: ./gradlew flywayMigrate
3. Regenerate jOOQ:           ./gradlew generateJooq
4. Update Kotlin code to use new fields
5. Compile -- if anything broke, compiler catches it
6. Commit migration + generated code + application code together
```

### Production Deployment Strategy

**1. Backward-compatible migrations.** We followed the "expand and contract" pattern:

```sql
-- Phase 1 (Deploy migration, old code still running):
-- ADD columns, ADD tables, ADD indexes -- never DROP or RENAME
ALTER TABLE tickets ADD COLUMN assignee_id BIGINT REFERENCES users(id);

-- Phase 2 (Deploy new code that writes to both old and new columns):
-- Application writes to both assigned_to and assignee_id

-- Phase 3 (After all services use new column):
-- Cleanup migration
ALTER TABLE tickets DROP COLUMN assigned_to;
```

**2. Zero-downtime migration rules:**
- Never add a NOT NULL column without a DEFAULT
- Never rename columns directly (add new, migrate data, drop old)
- Never drop columns that running code still references
- Add indexes CONCURRENTLY in PostgreSQL: `CREATE INDEX CONCURRENTLY`

```sql
-- V10__add_index_concurrently.sql
-- Flyway cannot run this in a transaction, so:
-- flyway.postgresql.transactional.lock = false
CREATE INDEX CONCURRENTLY idx_tickets_created_at ON tickets(created_at);
```

**3. Large data migrations were done asynchronously:**
```kotlin
// For millions of rows, we ran backfills as background jobs
@Scheduled(fixedDelay = 5000)
fun backfillAssigneeId() {
    val updated = dsl.update(TICKETS)
        .set(TICKETS.ASSIGNEE_ID, TICKETS.ASSIGNED_TO) // Copy from old to new column
        .where(TICKETS.ASSIGNEE_ID.isNull)
        .and(TICKETS.ASSIGNED_TO.isNotNull)
        .limit(1000) // Batch to avoid locking
        .execute()

    if (updated == 0) {
        log.info("Backfill complete")
        // Disable this scheduled task
    }
}
```

### Rollback Strategy

Flyway does not support automatic rollbacks in the community edition. Our approach:
- Every migration had a corresponding rollback script stored in a `rollback/` directory (not executed by Flyway, but available for manual use)
- For critical migrations, we took a database snapshot before deploying
- We tested migrations against a production-clone database before deploying to production

### Lessons Learned

1. **Always test migrations with production-scale data.** A migration that runs in 100ms on a dev database with 100 rows can lock a production table with 10M rows for minutes.
2. **Version migrations with the feature branch.** The migration script, jOOQ-generated code, and application code should all be in the same PR.
3. **Never edit a migration that has been applied in any environment.** Flyway checksums will fail. Instead, write a new migration.
