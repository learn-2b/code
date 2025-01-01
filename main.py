import os  # لإدارة الملفات والمجلدات
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # لتحميل الصور وتحضيرها

# 1. تحديد مسارات المجلدات
base_dir = "dataset"  # اسم المجلد الرئيسي للبيانات
train_dir = os.path.join(base_dir, "training")  # مسار مجلد التدريب
val_dir = os.path.join(base_dir, "validation")  # مسار مجلد التحقق
test_dir = os.path.join(base_dir, "test")       # مسار مجلد الاختبار

# 2. إعداد مولد بيانات التدريب
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # تطبيع الصور: تحويل القيم من 0-255 إلى 0-1
    rotation_range=20,        # تدوير الصور عشوائيًا
    width_shift_range=0.2,    # تحريك الصور أفقيًا
    height_shift_range=0.2,   # تحريك الصور عموديًا
    zoom_range=0.2,           # تكبير الصور عشوائيًا
    horizontal_flip=True      # عكس الصور أفقيًا
)

# 3. إعداد مولد بيانات التحقق والاختبار (بدون تغييرات على الصور)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)

# 4. إنشاء مولد بيانات التدريب
train_generator = train_datagen.flow_from_directory(
    train_dir,                 # مسار مجلد التدريب
    target_size=(224, 224),    # تغيير حجم الصور إلى 224x224
    batch_size=32,             # تحميل 32 صورة في كل دفعة
    class_mode='categorical'   # التصنيف إلى فئات متعددة
)

# 5. إنشاء مولد بيانات التحقق
val_generator = val_test_datagen.flow_from_directory(
    val_dir,                   # مسار مجلد التحقق
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 6. إنشاء مولد بيانات الاختبار
test_generator = val_test_datagen.flow_from_directory(
    test_dir,                  # مسار مجلد الاختبار
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 7. عرض أسماء الفئات التي تم تحميلها
print("Class Labels:", train_generator.class_indices)
