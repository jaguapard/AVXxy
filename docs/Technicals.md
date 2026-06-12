# Dispatch pipeline
The dispatcher uses a constexpr FeatureSet struct to build a list of available operation implementations, and walks through this list using a fixed preference order, taking the first valid implementation. This order usually follows newer = better (higher priority) logic. You can look up the current order in `Dispatcher.h`. You can add new instruction sets once they get released if you follow these rules:

- Create a struct templated on FeatureSet
- Add it to the order of the Dispatcher
- Implement the struct by declaring and defining eval function overloads. First argument of the function must always be an operation tag type. Look into `tags.h` for reference. Requires-clauses must be used to limit which implementations the dispatcher sees as valid