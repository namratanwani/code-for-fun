class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

    def have_birthday(self):
        self.age += 1
        print(f"Happy Birthday! I am now {self.age} years old.")

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def study(self):
        print(f"{self.name} is studying hard.")

class Teacher(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id

    def teach(self):
        print(f"{self.name} is teaching a class.")


person1 = Person("John", 25)
student1 = Student("Alice", 20, "S123")
teacher1 = Teacher("Mr. Smith", 35, "T789")


person1.introduce()
student1.introduce()
student1.study()

teacher1.introduce()
teacher1.teach()


person1.have_birthday()
person1.introduce()


print(f"{student1.name} has a student ID: {student1.student_id}")
print(f"{teacher1.name} has an employee ID: {teacher1.employee_id}")
