/*
Alumno: Mariana Paula Cáceres Urquizo
Curso: Computación Gráfica

Notas:
- Las teclas son para lo siguiente:
	- Esc: salir del programa
	- Q: aumentar el rojo del fondo
	- A: disminuir el rojo del fondo
	- W: aumentar el verde del fondo
	- S: disminuir el verde del fondo
	- E: aumentar el azul del fondo
	- D: disminuir el azul del fondo
	- 1: rotar X
	- 2: rotar Y
	- 3: rotar Z
	- 4: escalar X
	- 5: escalar Y
	- 6: escalar Z
	- Up, Down, Left, Right: mover
*/

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

template <unsigned int N, typename T>
class Vector
{
public:
	std::array<T, N> vals;

	Vector<N, T>(const std::array<T, N>& val)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = val[i];
		}
	}

	template <unsigned int M>
	Vector<N, T>(const Vector<M, T>& val, bool last);

	Vector<N, T>()
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = 0;
		}
	}

	T length2() const
	{
		T mod = vals[0] * vals[0];
		for (int i = 1; i < N; i++)
		{
			mod = mod + vals[i] * vals[i];
		}
		return mod;
	}
	
	T length() const
	{
		return std::sqrt(length2());
	}

	Vector<N, T> direction()
	{
		return *this / length();
	}

	void normalize()
	{
		operator/=(length());
	}

	T eucliDist2(const Vector<N, T>& rhs)
	{
		return (operator-(rhs)).length2();
	}

	T eucliDist(const Vector<N, T>& rhs)
	{
		return std::sqrt(eucliDist2(rhs));
	}

	Vector<N, T>& operator=(const Vector<N, float>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = rhs.vals[i];
		}

		return *this;
	}

	bool operator==(const Vector<N, T>& rhs) const
	{
		bool eq = true;
		for (int i = 0; i < N; i++)
		{
			eq = eq && vals[i] == rhs.vals[i];
		}
		return eq;
	}

	bool operator!=(const Vector<N, T>& rhs) const
	{
		return !(*this == rhs);
	}

	Vector<N, T> operator+ (const Vector<N, T>& rhs) const
	{
		Vector<N, T> tmp;
		for (int i = 0; i < N; i++)
		{
			tmp.vals[i] = vals[i] + rhs.vals[i];
		}
		return tmp;
	}

	Vector<N, T> operator- (const Vector<N, T>& rhs) const
	{
		Vector<N, T> tmp;
		for (int i = 0; i < N; i++)
		{
			tmp.vals[i] = vals[i] - rhs.vals[i];
		}
		return tmp;
	}

	Vector<N, T> operator* (const Vector<N, T>& rhs) const
	{
		Vector<N, T> tmp;
		for (int i = 0; i < N; i++)
		{
			tmp.vals[i] = vals[i] * rhs.vals[i];
		}
		return tmp;
	}

	Vector<N, T> operator* (T scale)
	{
		Vector<N, T> tmp;
		for (int i = 0; i < N; i++)
		{
			tmp.vals[i] = vals[i] * scale;
		}
		return tmp;
	}

	T operator& (const Vector<N, T>& rhs) const
	{
		T ans = vals[0] * rhs.vals[0];
		for (int i = 1; i < N; i++)
		{
			ans = ans + vals[i] * rhs.vals[i];
		}
		return ans;
	}

	Vector<N, T> operator/ (const Vector<N, T>& rhs) const
	{
		Vector<N, T> tmp;
		for (int i = 0; i < N; i++)
		{
			tmp.vals[i] = vals[i] / rhs.vals[i];
		}
		return tmp;
	}

	Vector<N, T> operator/ (T scale) const
	{
		Vector<N, T> tmp;
		for (int i = 0; i < N; i++)
		{
			tmp.vals[i] = vals[i] / scale;
		}
		return tmp;
	}

	void operator+= (const Vector<N, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = vals[i] + rhs.vals[i];
		}
	}

	void operator-= (const Vector<N, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = vals[i] - rhs.vals[i];
		}
	}

	void operator*= (const Vector<N, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = vals[i] * rhs.vals[i];
		}
	}

	void operator*= (T scale)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = vals[i] * scale;
		}
	}

	void operator/= (const Vector<N, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = vals[i] / rhs.vals[i];
		}
	}

	void operator/= (T scale)
	{
		for (int i = 0; i < N; i++)
		{
			vals[i] = vals[i] / scale;
		}
	}

	T& operator[](int indx)
	{
		return vals[indx];
	}

	const T& operator[](int indx) const
	{
		return vals[indx];
	}

	T& x = vals[0];
	T& y = vals[1];
};

template <unsigned int N, typename T>
template <unsigned int M>
Vector<N, T>::Vector(const Vector<M, T> &val, bool last)
{
	int rep = std::min(N, M);
	for (int i = 0; i < rep; i++)
	{
		vals[i] = val[i];
	}
	if (last)
		vals[N-1] = 1;
}

template <unsigned int N, unsigned int M, typename T>
class Matrix
{
public:
	std::array<std::array<T, M>, N> vals;

	Matrix<N, M, T>()
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = 0;
			}
		}
	}

	T* raw()
	{
		T *tmp = new T[N*M];

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp[i*M+j] = vals[i][j];
			}
		}

		return tmp;
	}

	Matrix<N, M, T>& operator=(const Matrix<N, M, T> mtrx)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = mtrx.vals[i][j];
			}
		}

		return *this;
	}

	Vector<N, T> operator&(const Vector<M, T>& rhs) const
	{
		Vector<N, T> ans;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				ans[i] += vals[i][j] * rhs[j];
			}
		}
		return ans;
	}

	Matrix<N, N, T> operator&(const Matrix<N, N, T>& rhs) const
	{
		Matrix<N, N, T> ans;

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < N; k++)
				{
					ans.vals[i][j] += vals[i][k] * rhs.vals[k][j];
				}
			}
		}

		return ans;
	}

	void operator&=(const Matrix<N, N, T>& rhs)
	{
		vals = operator&(rhs).vals;
	}

	bool operator==(const Matrix<N, M, T>& rhs) const
	{
		bool eq = true;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				eq = eq && vals[i][j] == rhs.vals[i][j];
			}
		}
		return eq;
	}

	bool operator!=(const Vector<N, T>& rhs) const
	{
		return !(*this == rhs);
	}

	Matrix<N, M, T> operator+ (const Matrix<N, M, T>& rhs) const
	{
		Matrix<N, M, T> tmp;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp.vals[j][i] = vals[i][j] + rhs.vals[i][j];
			}
		}
		return tmp;
	}

	Matrix<N, M, T> operator- (const Matrix<N, M, T>& rhs) const
	{
		Matrix<N, M, T> tmp;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp.vals[j][i] = vals[i][j] - rhs.vals[i][j];
			}
		}
		return tmp;
	}

	Matrix<N, M, T> operator* (const Matrix<N, M, T>& rhs) const
	{
		Matrix<N, M, T> tmp;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp.vals[j][i] = vals[i][j] * rhs.vals[i][j];
			}
		}
		return tmp;
	}

	Matrix<N, M, T> operator* (T scale) const
	{
		Matrix<N, M, T> tmp;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp.vals[j][i] = vals[i][j] * scale;
			}
		}
		return tmp;
	}

	Matrix<N, M, T> operator/ (const Matrix<N, M, T>& rhs) const
	{
		Matrix<N, M, T> tmp;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp.vals[j][i] = vals[i][j] / rhs.vals[i][j];
			}
		}
		return tmp;
	}

	Matrix<N, M, T> operator/ (T scale) const
	{
		Matrix<N, M, T> tmp;
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				tmp.vals[j][i] = vals[i][j] / scale;
			}
		}
		return tmp;
	}

	void operator+= (const Matrix<N, M, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = vals[i][j] + rhs.vals[i][j];
			}
		}
	}

	void operator-= (const Matrix<N, M, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = vals[i][j] - rhs.vals[i][j];
			}
		}
	}

	void operator*= (const Matrix<N, M, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = vals[i][j] * rhs.vals[i][j];
			}
		}
	}

	void operator*= (T scale)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = vals[i][j] * scale;
			}
		}
	}

	void operator/= (const Matrix<N, M, T>& rhs)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = vals[i][j] / rhs.vals[i][j];
			}
		}
	}

	void operator/= (T scale)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[j][i] = vals[i][j] / scale;
			}
		}
	}
	
	static Matrix<N, M, T> Identity()
	{
		Matrix<N, M, T> tmp;

		for (int i = 0; i < N; i++)
		{
			tmp.vals[i][i] = 1;
		}

		return tmp;
	}	
	
	static Matrix<N, M, T> Translation(const Vector<N, T>& trans)
	{
		Matrix<N, M, T> tmp = Identity();

		for (int i = 0; i < N; i++)
		{
			tmp.vals[i][M-1] = trans[i];
		}

		return tmp;
	}

	static Matrix<N, M, T> Scaling(const Vector<N, T>& scale)
	{
		Matrix<N, M, T> tmp;

		for (int i = 0; i < N; i++)
		{
			tmp.vals[i][i] = scale[i];
		}

		return tmp;
	}

	static Matrix<N, M, T> RotateX(T angle)
	{
		Matrix<N, M, T> tmp = Identity();

		tmp.vals[1][1] = std::cos(angle);
		tmp.vals[1][2] = -std::sin(angle);
		tmp.vals[2][1] = std::sin(angle);
		tmp.vals[2][2] = std::cos(angle);

		return tmp;
	}

	static Matrix<N, M, T> RotateY(T angle)
	{
		Matrix<N, M, T> tmp = Identity();

		tmp.vals[0][0] = std::cos(angle);
		tmp.vals[0][2] = std::sin(angle);
		tmp.vals[2][0] = -std::sin(angle);
		tmp.vals[2][2] = std::cos(angle);

		return tmp;
	}

	static Matrix<N, M, T> RotateZ(T angle)
	{
		Matrix<N, M, T> tmp = Identity();

		tmp.vals[0][0] = std::cos(angle);
		tmp.vals[0][1] = -std::sin(angle);
		tmp.vals[1][0] = std::sin(angle);
		tmp.vals[1][1] = std::cos(angle);

		return tmp;
	}
};

template<unsigned int N, typename T>
void printM(const Matrix<N, N, T>& matrx)
{
	std::cout << "---\n";
	for (auto v : matrx.vals)
	{
		for (auto x : v)
		{
			std::cout << x << ' ';
		}
		std::cout << '\n';
	}
	std::cout << "---\n";
}

unsigned int program;
float rgb[3] = {0.5f, 0.5f, 0.7f};

unsigned int compileShader(const char* shaderSource, GLenum shaderType);
unsigned int linkShaders(unsigned int, unsigned int);

const float pi = 3.1416f;

template <typename T>
class Object
{
protected:

	bool initialized = false;
	unsigned int vao;
	Vector<3, T> position;
	Matrix<4, 4, T> transformation = Matrix<4, 4, T>::Identity();

public:

	Object() {};

	~Object()
	{
		if (initialized)
			glDeleteVertexArrays(1, &vao);
	}

	virtual void draw() = 0;

	void transform(Matrix<4, 4, T> trans)
	{
		auto newPos = trans & Vector<4, T>({position[0], position[1], position[2], 1});
		position = Vector<3, T>({newPos[0], newPos	[1], newPos[2]});
		transformation = trans & transformation;
	}

	const Vector<3, T>& getPosition()
	{
		return position;
	}

};

typedef Vector<3, float> Vector3f;
typedef Vector<4, float> Vector4f;
typedef Matrix<4, 4, float> Matrix4f;

class CubePrimitive :
	public Object<float>
{
public:

	CubePrimitive(std::vector<int> colors, const Vector3f& offset)
	{
		while (colors.size() < 3)
		{
			colors.push_back(0);
		}

		int points[8] = {
			0b000,
			0b001,
			0b010,
			0b011,
			0b100,
			0b101,
			0b110,
			0b111
		};

		struct data {
			float positions[3];
			int color;
		};

		data vertices[24];
		unsigned int indices[36];

		int x = 1;
		for (int i = 0; i < 3; i++, x=x<<1)
		{
			int off1 = 0;
			int off0 = 0;
			for (int j = 0; j < 8; j++)
			{
				vertices[8*i+j].color = colors[i];
				if (points[j] & x)
				{
					vertices[8*i+off1].positions[0] = ((points[j] & 1) ? 0.5f : -0.5f) + offset[0];
					vertices[8*i+off1].positions[1] = ((points[j] & 2) ? 0.5f : -0.5f) + offset[1];
					vertices[8*i+off1].positions[2] = ((points[j] & 4) ? 0.5f : -0.5f) + offset[2];
					off1++;
				}
				else
				{
					vertices[8*i+4+off0].positions[0] = ((points[j] & 1) ? 0.5f : -0.5f) + offset[0];
					vertices[8*i+4+off0].positions[1] = ((points[j] & 2) ? 0.5f : -0.5f) + offset[1];
					vertices[8*i+4+off0].positions[2] = ((points[j] & 4) ? 0.5f : -0.5f) + offset[2];
					off0++;
				}
			}
		}

		for (int i = 0; i < 6; i++)
		{
			indices[6*i] = 4*i;
			indices[6*i+1] = 4*i+1;
			indices[6*i+2] = 4*i+2;
			indices[6*i+3] = 4*i+1;
			indices[6*i+4] = 4*i+2;
			indices[6*i+5] = 4*i+3;
		}

		// for (int i = 0; i < 24; i++)
		// {
		// 	std::cout << vertices[i].positions[0] << ' ' << vertices[i].positions[1] << ' ' << vertices[i].positions[2] << ' ' << vertices[i].color << '\n';
		// }

		unsigned int vbo, ebo;
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ebo);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 24*sizeof(data), vertices, GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36*sizeof(unsigned int), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(data), (void*)(offsetof(data, positions)));
		glEnableVertexAttribArray(0);
		glVertexAttribIPointer(1, 1, GL_INT, sizeof(data), (void*)(offsetof(data, color)));
		glEnableVertexAttribArray(1);

		initialized = true;
	}

	virtual void draw() override
	{
		glBindVertexArray(vao);

		unsigned int transformLoc = glGetUniformLocation(program, "transform");
		float *tmp = transformation.raw();
		// float tmp[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		glUniformMatrix4fv(transformLoc, 1, GL_TRUE, tmp);
		delete tmp;

		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
		// glDrawArrays(GL_LINE_LOOP, 0, 24);
	}
};

class Cube
{
	std::vector<CubePrimitive*> cubes;
	Vector3f position;

public:
	Cube()
	{
		std::cout << "Creating Cubes\n";
		int colorsx[3] = {1, 0, 5};
		int colorsy[3] = {2, 0, 4};
		int colorsz[3] = {3, 0, 6};
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					if (i == 1 && j == 1 && k == 1)
						continue;
				cubes.push_back(new CubePrimitive({colorsx[i], colorsy[j], colorsz[k]}, Vector3f({-1.05f+(float)i*1.05f, -1.05f+(float)j*1.05f, -1.05f+(float)k*1.05f})));
				}
			}
		}
		auto trans = Matrix4f::Scaling(Vector4f({0.5f, 0.5f, 0.5f, 1.0f}));
		transform(trans);

		std::cout << "Cubes created\n";
	}

	void transform(const Matrix4f& trans)
	{
		// auto newPos = trans & Vector4f({position[0], position[1], position[2], 1.0f});
		auto newPos = trans & Vector4f(position, true);
		position = Vector3f({newPos[0], newPos[1], newPos[2]}); 
		for (auto& cube : cubes)
		{
			cube->transform(trans);
		}
	}

	void draw()
	{
		for (auto& cube : cubes)
		{
			cube->draw();
		}
	}
};

Cube* cube;

void init()
{
	glClearColor(rgb[0], rgb[1], rgb[2], 1.0f);
	glEnable(GL_DEPTH_TEST); 

    const char *vertexShaderSource = "#version 330 core\n"
		"out vec4 outC;\n"
        "layout (location = 0) in vec3 aPos;\n"
		"layout (location = 1) in int c;\n"
		"uniform vec4 colors[7];\n"
		"uniform mat4 transform;\n"
        "void main()\n"
        "{\n"
		"	outC = colors[c];\n"
        "   gl_Position = transform * vec4(aPos, 1.0f);\n"
        "}\0";

	const char *fragmentShaderSource = "#version 330 core\n"
		"out vec4 FragColor;\n"
		"in vec4 outC;\n"
		"void main()\n"
		"{\n"
		"    FragColor = outC;\n"
		"}\0";
	
	unsigned int vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
	unsigned int fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
	program = linkShaders(vertexShader, fragmentShader);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	
	glPointSize(10);
	glLineWidth(5);

	cube = new Cube();
	cube->transform(Matrix4f::Scaling(Vector4f({0.5f, 0.5f, 0.5f, 1.0f})));

	float colors[] = {
		0.5f, 0.5f, 0.5f, 1.0f,
		0.0f, 0.6f, 0.3f, 1.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
		0.7f, 0.1f, 0.2f, 1.0f,
		1.0f, 0.8f, 0.0f, 1.0f,
		0.0f, 0.3f, 0.7f, 1.0f,
		1.0f, 0.3f, 0.0f, 1.0f
	};
	
	unsigned int colorsLoc = glGetUniformLocation(program, "colors");
	glUseProgram(program);
	glUniform4fv(colorsLoc, 7, colors);
}

void processKeyInput(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action != GLFW_PRESS)
		return;
    if (key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(window, true);
    }
	int check[6] = {GLFW_KEY_Q, GLFW_KEY_A, GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_E, GLFW_KEY_D};
	for (int i = 0; i < 3; i++)
	{
		if (key == check[2*i])
		{
			rgb[i] = std::min(1.0f, rgb[i] + 0.1f);
		}
		if (key == check[2*i+1])
		{
			rgb[i] = std::max(0.0f, rgb[i] - 0.1f);
		}
	}

	auto transformtrx = Matrix4f::Identity();
	// printM(transformtrx);

	if (key == GLFW_KEY_1)
		transformtrx &= Matrix4f::RotateX(1.0f);
	if (key == GLFW_KEY_2)
		transformtrx &= Matrix4f::RotateY(1.0f);
	if (key == GLFW_KEY_3)
		transformtrx &= Matrix4f::RotateZ(1.0f);
	if (key == GLFW_KEY_UP)
	 	transformtrx &= Matrix4f::Translation(Vector4f({0.0f, 0.1f, 0.0f, 1.0f}));
	if (key == GLFW_KEY_DOWN)
		transformtrx &= Matrix4f::Translation(Vector4f({0.0f, -0.1f, 0.0f, 1.0f}));
	if (key == GLFW_KEY_LEFT)
		transformtrx &= Matrix4f::Translation(Vector4f({-0.1f, 0.0f, 0.0f, 1.0f}));
	if (key == GLFW_KEY_RIGHT)
		transformtrx &= Matrix4f::Translation(Vector4f({0.1f, 0.0f, 0.0f, 1.0f}));
	if (key == GLFW_KEY_4)
		transformtrx &= Matrix4f::Scaling(Vector4f({1.1f, 1.0f, 1.0f, 1.0f}));
	if (key == GLFW_KEY_5)
		transformtrx &= Matrix4f::Scaling(Vector4f({1.0f, 1.1f, 1.0f, 1.0f}));
	if (key == GLFW_KEY_6)
		transformtrx &= Matrix4f::Scaling(Vector4f({1.0f, 1.0f, 1.1f, 1.0f}));

	// printM(transformtrx);
	cube->transform(transformtrx);
	// std::cout << cube->position[0] << ' ' << cube->position[1] << ' ' << cube->position[2] << '\n';

	glClearColor(rgb[0], rgb[1], rgb[2], 1.0f);
}

void render(GLFWwindow *window)
{	
	cube->draw();
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Window", NULL, NULL);

    if (!window)
    {
        std::cout << "Error creating window.\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGL(glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD.\n";
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, processKeyInput);
    
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

    init();
    std::cout << "Render Loop...\n";
    while(!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        render(window);

        glfwPollEvents();
        glfwSwapBuffers(window);
    }

    glfwTerminate();

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

unsigned int compileShader(const char* shaderSource, GLenum shaderType)
{
	unsigned int shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSource, NULL);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "Error compiling vertex shader.\n" << infoLog << '\n';
    }
	
	return shader;
}

unsigned int linkShaders(unsigned int vertexShader, unsigned int fragmentShader)
{
	unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

	int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "Error linking shaders.\n" << infoLog << '\n';
    }
	
	return shaderProgram;
}