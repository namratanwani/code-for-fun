class Person:
    def __init__(self, name, age):
        """
        sets properties `name` and `age`.

        Args:
            name (str): person's name that is assigned to the instance of the class.
            age (int): 21-year-old's age at the time of the function's call.

        """
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

    def have_birthday(self):
        """
        updates the age of an object and displays a message with the new age.

        """
        self.age += 1
        print(f"Happy Birthday! I am now {self.age} years old.")

class Student(Person):
    def __init__(self, name, age, student_id):
        """
        initializes a Student object by setting the name, age, and student ID
        attributes from arguments provided during object creation.

        Args:
            name (str): person's name.
            age (int): 2D student's age during initialization of the `Student`
                object in the function `__init__`.
            student_id (int): unique identification number assigned to each student.

        """
        super().__init__(name, age)
        self.student_id = student_id

    def study(self):
        print(f"{self.name} is studying hard.")

class Teacher(Person):
    def __init__(self, name, age, employee_id):
        """
        initializes an instance of a class, setting properties `name`, `age`, and
        `employee_id`.

        Args:
            name (str): person's name in the initialization of an Employee object.
            age (int): 25-year-old employee's age.
            employee_id (int): unique identifier of the employee being created and
                is used to store the value within the object instance, as passed
                to the parent class `__init__()` method.

        """
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
