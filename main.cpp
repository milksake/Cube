/*
Alumno: M. P. Cáceres Urquizo
Curso: Computación Gráfica

Notas:
- Las teclas son para lo siguiente:
	- Esc: salir del programa
	- Q: rotar cara izquierda
	- W: rotar cara derecha
	- E: rotar cara arriba
	- R: rotar cara abajo
	- T: rotar cara frente
	- Y: rotar cara posterior
	- 7: camara
	- 8: camara
	- 9: camara
*/

// #define _GLIBCXX_DEBUG 1

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
#include <map>

// Solver
#include <min2phase/min2phase.h>
#include <min2phase/tools.h>

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
	Vector<N, T>(const Vector<M, T>& val, bool last = false);

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

	Vector<N, T> operator* (T scale) const
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

	Vector<N, T> operator% (const Vector<N, T>& rhs) const;

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

	void transpose()
	{
		auto old = *this;

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < M; j++)
			{
				vals[i][j] = old.vals[j][i];
			}
		}
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

	static Matrix<N, M, T> FromBase(const Vector<N, T>& u, const Vector<N, T>& v, const Vector<N, T>& w)
	{
		Matrix<N, M, T> tmp;

		int lim = std::min(M, 3u);
		
		const Vector<N, T> * ptrs[3] = {&u, &v, &w};
		for (int j = 0; j < lim; j++)
		{
			for (int i = 0; i < N; i++)
			{
				tmp.vals[i][j] = (*ptrs[j])[i];
			}
		}

		return tmp;
	}

	static Matrix<N, M, T> Dual(const Vector<N, T>& vec)
	{
		Matrix<N, M, T> tmp;

		tmp.vals[0][1] = -vec[2];
		tmp.vals[1][0] = vec[2];
		tmp.vals[2][0] = -vec[1];
		tmp.vals[0][2] = vec[1];
		tmp.vals[1][2] = -vec[0];
		tmp.vals[2][1] = vec[0];

		return tmp;		
	}

	static Matrix<N, M, T> LookAt(const Vector<N, T>& pos, const Vector<N, T>& center, const Vector<N, T>& up)
	{
		auto w = pos - center;
		w.normalize();

		auto u = up % w;
		u.normalize();

		auto v = w % u;

		auto rotation = Matrix<N, M, T>::FromBase(u, v, w);
		rotation.transpose();

		auto ret = rotation & Matrix<N, M, T>::Translation(Vector<4, T>(pos * -1.0));
		if (N >= 4 && M >= 4)
			ret.vals[3][3] = (T)1;

		return ret;
	}

	static Matrix<N, M, T> Ortho(T left, T right, T bottom, T top, T near, T far)
	{
		auto tmp = Matrix<N, M, T>::Scaling(Vector<N, T>({(T)2/(right-left), (T)2/(top-bottom), (T)2/(far-near), (T)1})) & Matrix<N, M, T>::Translation(Vector<N, T>({-(left+right)/(T)2, -(top+bottom)/(T)2, -(near+far)/(T)2, (T)1}));
		tmp.vals[2][2] *= -1;

		return tmp;
	}

	static Matrix<N, M, T> Perspec(T fov, T aspect, T zNear, T zFar)
	{
		T height = zNear * std::tan(fov / 2.0f);
		T width = height * aspect;

		T l = -width, r = width, b = -height, t = height;
		T n = zNear, f = zFar;

		Matrix<N, M, T> tmp;

		tmp.vals[0][0] = 2.f*n/(r-l);

		tmp.vals[1][1] = 2.*n/(t-b);

		tmp.vals[0][2] = (r+l)/(r-l);
		tmp.vals[1][2] = (t+b)/(t-b);
		tmp.vals[2][2] = -(f+n)/(f-n);
		tmp.vals[3][2] = -1.f;

		tmp.vals[2][3] = -2.f*(f*n)/(f-n);

		return tmp;
	}
};

template<unsigned int N, typename T>
Vector<N, T> Vector<N, T>::operator% (const Vector<N, T>& rhs) const
{
	return Vector<N, T>(Matrix<3, 3, T>::Dual(Vector<3, T>(*this)) & Vector<3, T>(rhs));
}

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
		position = Vector<3, T>(trans & Vector<4, T>(position, true));
		transformation = trans & transformation;
	}

	const Vector<3, T>& getPosition()
	{
		return position;
	}

};

template<typename T>
class Animation
{
	Matrix<4, 4, T> transformation;
	int currFrames = 0;
	int maxFrames = 0;
	bool finished;
	std::vector<Object<T>*> objects;

public:
	Animation() :
		finished(true) {}

	Animation(const Matrix<4, 4, T>& trans, int duration, const std::vector<Object<T>*>& objs):
		transformation(trans), maxFrames(duration), objects(objs), finished(false) {}

	bool run()
	{
		if (finished)
			return false;
		for (auto& obj : objects)
		{
			obj->transform(transformation);
		}
		currFrames++;
		if (currFrames >= maxFrames)
			finished = true;
		
		return true;
	}
};

typedef Vector<3, float> Vector3f;
typedef Vector<4, float> Vector4f;
typedef Matrix<4, 4, float> Matrix4f;

class CubePrimitive :
	public Object<float>
{
public:

	CubePrimitive(std::vector<int> colors, const Matrix4f& iniTrans)
	{
		transformation = iniTrans;

		while (colors.size() < 6)
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
				vertices[8*i+j].color = (j < 4) ? colors[2*i] : colors[2*i+1];
				if (points[j] & x)
				{
					vertices[8*i+off1].positions[0] = ((points[j] & 1) ? 0.5f : -0.5f);
					vertices[8*i+off1].positions[1] = ((points[j] & 2) ? 0.5f : -0.5f);
					vertices[8*i+off1].positions[2] = ((points[j] & 4) ? 0.5f : -0.5f);
					off1++;
				}
				else
				{
					vertices[8*i+4+off0].positions[0] = ((points[j] & 1) ? 0.5f : -0.5f);
					vertices[8*i+4+off0].positions[1] = ((points[j] & 2) ? 0.5f : -0.5f);
					vertices[8*i+4+off0].positions[2] = ((points[j] & 4) ? 0.5f : -0.5f);
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
	std::array<std::array<std::array<int, 3>, 3>, 3> indices;
	Vector3f position;
	int animationTime = 200;
	Animation<float> animation;
	std::vector<char> solving;
	int solvingIndex = -1;

	std::map<char, std::array<int, 3>> moves = {
		{'B', {2, 0, 1}},
		{'F', {2, 2, 1}},
		{'L', {0, 0, 1}},
		{'R', {0, 2, 1}},
		{'D', {1, 0, 1}},
		{'U', {1, 2, 1}},
		{'b', {2, 0, -1}},
		{'f', {2, 2, -1}},
		{'l', {0, 0, -1}},
		{'r', {0, 2, -1}},
		{'d', {1, 0, -1}},
		{'u', {1, 2, -1}},
		{'X', {0, 1, 1}},
		{'Y', {1, 1, 1}},
		{'Z', {2, 1, 1}},
		{'x', {0, 1, -1}},
		{'y', {1, 1, -1}},
		{'z', {2, 1, -1}},
	};

	std::map<char, std::vector<std::vector<char>>> colors;

	bool modifiable = true;

	void setUpSolverAnimation(const std::vector<char>& moves)
	{
		solving = moves;
		solvingIndex = 0;
	}

public:
	Cube()
	{
		std::cout << "Creating Cubes...\n";

		colors['U'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'U'));
		colors['D'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'D'));
		colors['L'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'L'));
		colors['R'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'R'));
		colors['F'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'F'));
		colors['B'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'B'));

		indices[1][1][1] = -1;

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
					
					indices[i][j][k] = cubes.size();

					auto iniTrans =  Matrix4f::Translation(Vector4f({-1.05f+(float)i*1.05f, -1.05f+(float)j*1.05f, -1.05f+(float)k*1.05f, 1.0f}));
					cubes.push_back(new CubePrimitive({
						colorsx[i] * (i == 2), colorsx[i] * (i == 0),
						colorsy[j] * (j == 2), colorsy[j] * (j == 0),
						colorsz[k] * (k == 2), colorsz[k] * (k == 0)
					}, iniTrans));
				}
			}
		}
		auto trans = Matrix4f::Scaling(Vector4f({0.5f, 0.5f, 0.5f, 1.0f}));
		transform(trans);

		// std::cout << "---Colors\n";
		// for (auto& col : colors)
		// {
		// 	std::cout << '\t' << col.first << '\n';
		// 	for (int i = 0; i < 3; i++)
		// 	{
		// 		for (int j = 0; j < 3; j++)
		// 		{
		// 			std::cout << col.second[i][j] << ' ';
		// 		}
		// 		std::cout << '\n';
		// 	}
		// }
		// std::cout << "-----\n";

		std::cout << "Cubes created\n";
	}

	void transform(const Matrix4f& trans)
	{
		if (!modifiable)
			return;

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
		bool tmp = !animation.run();
		if (solvingIndex == -1)
		{
			modifiable = tmp;
		}
		else if (solvingIndex < solving.size())
		{
			if (tmp)
			{
				modifiable = true;
				Move(solving[solvingIndex]);
				solvingIndex++;
			}
		}
		else
		{
			solvingIndex = -1;
			modifiable = tmp;

			colors['U'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'U'));
			colors['D'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'D'));
			colors['L'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'L'));
			colors['R'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'R'));
			colors['F'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'F'));
			colors['B'] = std::vector<std::vector<char>>(3, std::vector<char>(3, 'B'));
		}
		for (auto& cube : cubes)
		{
			cube->draw();
		}
	}

	void Move(const std::array<int, 3>& arr)
	{
		if (!modifiable)
			return;

		std::vector<Object<float>*> objs;

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				int ind;
				if (arr[0] == 0)
					ind = indices[arr[1]][i][j];
				else if (arr[0] == 1)
					ind = indices[i][arr[1]][j];
				else
					ind = indices[i][j][arr[1]];
				
				if (ind == -1)
					continue;
				
				objs.push_back(cubes[ind]);
			}
		}

		if (arr[0] == 0)
		{
			animation = Animation<float>(Matrix4f::RotateX(pi*arr[2]/2 / animationTime), animationTime, objs);
			
			std::vector<char> order = {'U', 'F', 'D', 'B'};
			if (arr[2] > 0)
				reverse(order.begin(), order.end());
			int ind1, ind2 = (order[0]=='B')?(3-arr[1]-1):arr[1];
			char tmp[3];
			for (int i = 0; i < 3; i++)
			{
				ind1 = (order[0]=='B') ? 3-i-1 : i;
				tmp[i] = colors[order[0]][ind1][ind2];
			}
			for (int j = 0; j < order.size()-1; j++)
			{
				ind2 = (order[j]=='B')?(3-arr[1]-1):arr[1];
				int ind22 = (order[j+1]=='B')?(3-arr[1]-1):arr[1];
				for (int i = 0; i < 3; i++)
				{
					ind1 = (order[j]=='B') ? 3-i-1 : i;
					int ind11 = (order[j+1]=='B') ? 3-i-1 : i;
					colors[order[j]][ind1][ind2] = colors[order[j+1]][ind11][ind22];
				}
			}
			ind2 = (order[order.size()-1]=='B')?(3-arr[1]-1):arr[1];
			for (int i = 0; i < 3; i++)
			{
				ind1 = (order[order.size()-1]=='B') ? (3-i-1) : i;
				colors[order[order.size()-1]][ind1][ind2] = tmp[i];
			}

			if (arr[1] != 1)
			{
				auto& face = colors[(arr[1] == 0) ? 'L': 'R'];

				if ((arr[2] > 0 && arr[1] == 0) || (arr[2] < 0 && arr[1] == 2))
				{
					for (int i = 0; i < 3 / 2; i++)
					{
						for (int j = i; j < 3 - i - 1; j++)
						{
							int temp = face[i][j];
							face[i][j] = face[3 - 1 - j][i];
							face[3 - 1 - j][i] = face[3 - 1 - i][3 - 1 - j];
							face[3 - 1 - i][3 - 1 - j] = face[j][3 - 1 - i];
							face[j][3 - 1 - i] = temp;
						}
					}
				}
				else
				{
					for (int i = 0; i < 3 / 2; i++)
					{
						for (int j = i; j < 3 - i - 1; j++)
						{
							int temp = face[i][j];
							face[i][j] = face[j][3 - 1 - i];
							face[j][3 - 1 - i] = face[3 - 1 - i][3 - 1 - j];
							face[3 - 1 - i][3 - 1 - j] = face[3 - 1 - j][i];
							face[3 - 1 - j][i] = temp;
						}
					}
				}
			}
		}
		else if (arr[0] == 1)
		{
			animation = Animation<float>(Matrix4f::RotateY(pi*arr[2]/2 / animationTime), animationTime, objs);
		
			std::vector<char> order = {'F', 'R', 'B', 'L'};
			if (arr[2] > 0)
				reverse(order.begin(), order.end());
			int ind1, ind2 = 3-1-arr[1];
			char tmp[3];
			for (int i = 0; i < 3; i++)
			{
				ind1 = i;
				tmp[i] = colors[order[0]][ind2][ind1];
			}
			for (int j = 0; j < order.size()-1; j++)
			{
				ind2 = 3-1-arr[1];
				int ind22 = 3-1-arr[1];
				for (int i = 0; i < 3; i++)
				{
					ind1 = i;
					int ind11 = i;
					colors[order[j]][ind2][ind1] = colors[order[j+1]][ind22][ind11];
				}
			}
			ind2 = 3-1-arr[1];
			for (int i = 0; i < 3; i++)
			{
				ind1 = i;
				colors[order[order.size()-1]][ind2][ind1] = tmp[i];
			}

			if (arr[1] != 1)
			{
				auto& face = colors[(arr[1] == 0) ? 'D': 'U'];

				if ((arr[2] < 0 && arr[1] == 2) || (arr[2] > 0 && arr[1] == 0))
				{
					for (int i = 0; i < 3 / 2; i++)
					{
						for (int j = i; j < 3 - i - 1; j++)
						{
							int temp = face[i][j];
							face[i][j] = face[3 - 1 - j][i];
							face[3 - 1 - j][i] = face[3 - 1 - i][3 - 1 - j];
							face[3 - 1 - i][3 - 1 - j] = face[j][3 - 1 - i];
							face[j][3 - 1 - i] = temp;
						}
					}
				}
				else
				{
					for (int i = 0; i < 3 / 2; i++)
					{
						for (int j = i; j < 3 - i - 1; j++)
						{
							int temp = face[i][j];
							face[i][j] = face[j][3 - 1 - i];
							face[j][3 - 1 - i] = face[3 - 1 - i][3 - 1 - j];
							face[3 - 1 - i][3 - 1 - j] = face[3 - 1 - j][i];
							face[3 - 1 - j][i] = temp;
						}
					}
				}
			}
		}
		else
		{
			animation = Animation<float>(Matrix4f::RotateZ(pi*arr[2]/2 / animationTime), animationTime, objs);
		

		}

		// std::cout << "---Colors\n";
		// for (auto& col : colors)
		// {
		// 	std::cout << '\t' << col.first << '\n';
		// 	for (int i = 0; i < 3; i++)
		// 	{
		// 		for (int j = 0; j < 3; j++)
		// 		{
		// 			std::cout << col.second[i][j] << ' ';
		// 		}
		// 		std::cout << '\n';
		// 	}
		// }
		// std::cout << "-----\n";
		
		modifiable = false;

		switch (arr[2])
		{
			case -2:
				break;
			case -1:
			{
				for (int i = 0; i < 3 / 2; i++)
				{
					for (int j = i; j < 3 - i - 1; j++)
					{
						if (arr[0] == 0)
						{
							int temp = indices[arr[1]][i][j];
							indices[arr[1]][i][j] = indices[arr[1]][3 - 1 - j][i];
							indices[arr[1]][3 - 1 - j][i] = indices[arr[1]][3 - 1 - i][3 - 1 - j];
							indices[arr[1]][3 - 1 - i][3 - 1 - j] = indices[arr[1]][j][3 - 1 - i];
							indices[arr[1]][j][3 - 1 - i] = temp;
						}
						else if (arr[0] == 1)
						{
							int temp = indices[j][arr[1]][i];
							indices[j][arr[1]][i] = indices[i][arr[1]][3 - 1 - j];
							indices[i][arr[1]][3 - 1 - j] = indices[3 - 1 - j][arr[1]][3 - 1 - i];
							indices[3 - 1 - j][arr[1]][3 - 1 - i] = indices[3 - 1 - i][arr[1]][j];
							indices[3 - 1 - i][arr[1]][j] = temp;
						}
						else
						{
							int temp = indices[i][j][arr[1]];
							indices[i][j][arr[1]] = indices[3 - 1 - j][i][arr[1]];
							indices[3 - 1 - j][i][arr[1]] = indices[3 - 1 - i][3 - 1 - j][arr[1]];
							indices[3 - 1 - i][3 - 1 - j][arr[1]] = indices[j][3 - 1 - i][arr[1]];
							indices[j][3 - 1 - i][arr[1]] = temp;
						}
					}
				}
			}
			break;
			case 1:
			{
				for (int i = 0; i < 3 / 2; i++)
				{
					for (int j = i; j < 3 - i - 1; j++)
					{
						if (arr[0] == 0)
						{
							int temp = indices[arr[1]][i][j];
							indices[arr[1]][i][j] = indices[arr[1]][j][3 - 1 - i];
							indices[arr[1]][j][3 - 1 - i] = indices[arr[1]][3 - 1 - i][3 - 1 - j];
							indices[arr[1]][3 - 1 - i][3 - 1 - j] = indices[arr[1]][3 - 1 - j][i];
							indices[arr[1]][3 - 1 - j][i] = temp;
						}
						else if (arr[0] == 1)
						{
							int temp = indices[j][arr[1]][i];
							indices[j][arr[1]][i] = indices[3 - 1 - i][arr[1]][j];
							indices[3 - 1 - i][arr[1]][j] = indices[3 - 1 - j][arr[1]][3 - 1 - i];
							indices[3 - 1 - j][arr[1]][3 - 1 - i] = indices[i][arr[1]][3 - 1 - j];
							indices[i][arr[1]][3 - 1 - j] = temp;
						}
						else
						{
							int temp = indices[i][j][arr[1]];
							indices[i][j][arr[1]] = indices[j][3 - 1 - i][arr[1]];
							indices[j][3 - 1 - i][arr[1]] = indices[3 - 1 - i][3 - 1 - j][arr[1]];
							indices[3 - 1 - i][3 - 1 - j][arr[1]] = indices[3 - 1 - j][i][arr[1]];
							indices[3 - 1 - j][i][arr[1]] = temp;
						}
					}
				}
			}
			break;
			default:
				break;

		}
	}
	
	void Move(char s)
	{
		if (!modifiable)
			return;

		auto it = moves.find(s);
		if (it == moves.end())
			return;
		Move(it->second);
	}

	void displace(int i)
	{
		i = i % 26;
		cubes[i]->transform(Matrix4f::Translation(Vector4f({0.0f, 0.0f, 1.0f, 1.0f})));
	}

	void solve()
	{
		if (!modifiable || solvingIndex != -1)
			return;

		modifiable = false;
		std::vector<char> order = {'U', 'R', 'F', 'D', 'L', 'B'};

		std::string facelet;
		for (char o : order)
		{
			auto& face = colors[o];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					facelet.push_back(face[i][j]);
				}
			}
		}

		std::cout << "Current State: " << facelet << '\n';
		uint8_t movesUsed;
		std::string solution = min2phase::solve(facelet, 21, 1000000, 0, 0, &movesUsed);
		std::cout << "Solution: " << solution << '\n';

		std::vector<char> solutionVector;
		for (size_t i = 0; i < solution.size(); ++i)
		{
			if (solution[i] != ' ')
			{
				if (solution[i] == '2')
				{
					solutionVector.push_back(solution[i-1]);
					solutionVector.push_back(solution[i-1]);
				}
				else if (solution[i] == '\'')
				{
					solutionVector.push_back(tolower(solution[i-1]));
				}
				else if (solution[i+1] != '2' && solution[i+1] != '\'')
				{
					solutionVector.push_back(solution[i]);
				}
			}
		}

		for (char& c: solutionVector)
		{
			if (c == 'R')
				c = 'r';
			else if (c == 'r')
				c = 'R';
			else if (c == 'U')
				c = 'u';
			else if (c == 'u')
				c = 'U';
			else if (c == 'F')
				c = 'f';
			else if (c == 'f')
				c = 'F';
		}
    
		setUpSolverAnimation(solutionVector);
	}
};

class Camera
{
	Vector3f pos;
	Vector3f center;
	Vector3f up;

	std::array<float, 6> limitsOrtho;
	std::array<float, 4> limitsPerspec;

	void updateCamera()
	{
		glUseProgram(program);

		auto tmp = Matrix4f::LookAt(Vector4f(pos), Vector4f(center), Vector4f(up));
		float *tmpptr = tmp.raw();
		// printM(tmp);

		unsigned int viewLoc = glGetUniformLocation(program, "view");
		glUniformMatrix4fv(viewLoc, 1, GL_TRUE, tmpptr);

		delete tmpptr;
	}

	void updateLimitsOrtho()
	{
		glUseProgram(program);

		auto tmp = Matrix4f::Ortho(limitsOrtho[0], limitsOrtho[1], limitsOrtho[2], limitsOrtho[3], limitsOrtho[4], limitsOrtho[5]);
		float* tmpptr = tmp.raw();
		// printM(tmp);

		unsigned int perpecLoc = glGetUniformLocation(program, "projection");
		glUniformMatrix4fv(perpecLoc, 1, GL_TRUE, tmpptr);

		delete tmpptr;
	}
	
	void updateLimitsPerspec()
	{
		glUseProgram(program);

		auto tmp = Matrix4f::Perspec(limitsPerspec[0], limitsPerspec[1], limitsPerspec[2], limitsPerspec[3]);
		float* tmpptr = tmp.raw();
		// printM(tmp);

		unsigned int perpecLoc = glGetUniformLocation(program, "projection");
		glUniformMatrix4fv(perpecLoc, 1, GL_TRUE, tmpptr);

		delete tmpptr;
	}

public:
	bool camera;

	Camera(const Vector3f& _pos, const Vector3f& _center, const Vector3f& _up, const std::array<float, 6>& ortho, const std::array<float, 4>& perspec, bool cam) :
		camera(cam)
	{
		pos = _pos;
		center = _center;
		up = _up;
		limitsOrtho = ortho;
		limitsPerspec = perspec;

		updateCamera();
		if (cam)
			updateLimitsPerspec();
		else
			updateLimitsOrtho();
	}

	Vector3f getPosition() const
	{
		return pos;
	}

	void setPosition(const Vector3f& vec)
	{
		pos = vec;

		updateCamera();
	}

	Vector3f getCenter() const
	{
		return center;
	}

	void setCenter(const Vector3f& vec)
	{
		center = vec;

		updateCamera();
	}

	Vector3f getUpVector() const
	{
		return up;
	}

	void setUpVector(const Vector3f& vec)
	{
		up = vec;

		updateCamera();
	}

	std::array<float, 6> getLimitsOrtho() const
	{
		return limitsOrtho;
	}

	void setLimitsOrtho(const std::array<float, 6>& arr)
	{
		limitsOrtho = arr;

		updateLimitsOrtho();
	}

	std::array<float, 4> getLimitsPerspec() const
	{
		return limitsPerspec;
	}

	void setLimitsPerspec(const std::array<float, 4>& arr)
	{
		limitsPerspec = arr;

		updateLimitsPerspec();
	}

	void move(const Vector3f& offset)
	{
		pos += offset;
		center += offset;

		updateCamera();
	}

	void changePerspective()
	{
		camera = !camera;
		if (camera)
			updateLimitsPerspec();
		else
			updateLimitsOrtho();
	}
};

Camera* camera;

Cube* cube;

int disp = 0;

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
		"uniform mat4 view;\n"
		"uniform mat4 projection;\n"
        "void main()\n"
        "{\n"
		"	outC = colors[c];\n"
        "   gl_Position = projection * view * transform * vec4(aPos, 1.0f);\n"
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
		0.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 0.6f, 0.3f, 1.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
		0.7f, 0.1f, 0.2f, 1.0f,
		1.0f, 0.8f, 0.0f, 1.0f,
		0.0f, 0.3f, 0.7f, 1.0f,
		1.0f, 0.3f, 0.0f, 1.0f
	};

	glUseProgram(program);
	
	unsigned int colorsLoc = glGetUniformLocation(program, "colors");
	glUniform4fv(colorsLoc, 7, colors);

	camera = new Camera(
		Vector3f({0.2f, 0.2f, 2.5f}),
		Vector3f({0.0f, 0.0f, 0.0f}),
		Vector3f({0.0f, 1.0f, 0.0f}),
		{-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 5.0f},
		{pi/4, 800.0f/600.0f, 0.1f, 5.0f},
		true
	);
}

void processKeyInput(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action != GLFW_PRESS)
		return;
    if (key == GLFW_KEY_ESCAPE)
    {
        glfwSetWindowShouldClose(window, true);
    }
	// int check[6] = {GLFW_KEY_Q, GLFW_KEY_A, GLFW_KEY_W, GLFW_KEY_S, GLFW_KEY_E, GLFW_KEY_D};
	// for (int i = 0; i < 3; i++)
	// {
	// 	if (key == check[2*i])
	// 	{
	// 		rgb[i] = std::min(1.0f, rgb[i] + 0.1f);
	// 	}
	// 	if (key == check[2*i+1])
	// 	{
	// 		rgb[i] = std::max(0.0f, rgb[i] - 0.1f);
	// 	}
	// }

	//auto transformtrx = Matrix4f::Identity();
	// printM(transformtrx);
	/*
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
		transformtrx &= Matrix4f::Scaling(Vector4f({1.0f, 1.0f, 1.1f, 1.0f}));*/
	if (key == GLFW_KEY_7)
	{
		camera->setPosition(Vector3f(Matrix4f::RotateX(0.1f) & Vector4f(camera->getPosition())));
	}
	if (key == GLFW_KEY_8)
	{
		camera->setPosition(Vector3f(Matrix4f::RotateY(0.1f) & Vector4f(camera->getPosition())));
	}
	if (key == GLFW_KEY_9)
	{
		camera->setPosition(Vector3f(Matrix4f::RotateZ(0.1f) & Vector4f(camera->getPosition())));
	}
	
	if (key == GLFW_KEY_Q)
		cube->Move('L');
	if (key == GLFW_KEY_W)
		cube->Move('X');
	if (key == GLFW_KEY_E)
		cube->Move('R');
	if (key == GLFW_KEY_R)
		cube->Move('D');
	if (key == GLFW_KEY_T)
		cube->Move('Y');
	if (key == GLFW_KEY_Y)
		cube->Move('U');
	if (key == GLFW_KEY_A)
		cube->Move('l');
	if (key == GLFW_KEY_S)
		cube->Move('x');
	if (key == GLFW_KEY_D)
		cube->Move('r');
	if (key == GLFW_KEY_F)
		cube->Move('d');
	if (key == GLFW_KEY_G)
		cube->Move('y');
	if (key == GLFW_KEY_H)
		cube->Move('u');
	
	if (key == GLFW_KEY_M)
		camera->changePerspective();

	if (key == GLFW_KEY_P)
		cube->solve();
	//if (key == GLFW_KEY_0)
	//{
	//	cube->displace(disp);
	//	disp++;
	//}

	// printM(transformtrx);
	// cube->transform(transformtrx);
	// std::cout << cube->position[0] << ' ' << cube->position[1] << ' ' << cube->position[2] << '\n';

	// glClearColor(rgb[0], rgb[1], rgb[2], 1.0f);
}

void render(GLFWwindow *window)
{	
	cube->draw();
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

int main()
{
	// Solver
	std::cout << "Initializing solver\n";
	min2phase::tools::setRandomSeed(time(nullptr));
	min2phase::init();
    // min2phase::loadFile("coords.m2pc");
	std::cout << "Solver initialized\n";

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