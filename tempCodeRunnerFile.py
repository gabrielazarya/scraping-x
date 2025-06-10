y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))