#include <array>
#include <cmath>
#include <fstream>
#include <vector>
#include "include/vec3.h"
#include "include/array.h"

#include <celerity.h>
#ifdef __CUDACC__
#define MSL_MANAGED __managed__
#define MSL_CONSTANT __constant__
#else
#define MSL_MANAGED
#define MSL_CONSTANT
#endif
using DataT = float;

const int FLAG_OBSTACLE = 1 << 0;
const int FLAG_KEEP_VELOCITY = 1 << 1;

typedef struct {
	unsigned int mantissa : 23;
	unsigned int exponent : 8;
	unsigned int sign : 1;
} floatparts;

const size_t Q = 19;
MSL_MANAGED float cellwidth = 1.0f;

typedef array<float, Q> cell_t;
typedef vec3<float> vec3f;

std::ostream& operator<< (std::ostream& os, const cell_t f) {
	os << "(" << f[0] << ", " << f[1] << ", " << f[2] << "...)";
	return os;
}
MSL_MANAGED vec3<int> size;

MSL_MANAGED float deltaT = 1.f;

MSL_MANAGED float tau = 0.65;

MSL_CONSTANT const array<vec3f, Q> offsets {
	{{0, 0, 0},
        {-1, 0, 0},
	    {1, 0, 0},
        {0, -1, 0},
	    {0, 1, 0},
        {0, 0, -1},
	    {0, 0, 1},
        {-1, -1, 0},
	    {-1, 1, 0},
	    {1, -1, 0},
        {1, 1, 0},
	    {-1, 0, -1},
        {-1, 0, 1},
	    {1, 0, -1},
        {1, 0, 1},
	    {0, -1, -1},
        {0, -1, 1},
	    {0, 1, -1},
	    {0, 1, 1}
    }
};

MSL_CONSTANT const array<unsigned char, Q> opposite = {
    0,
    2, 1, 4, 3, 6, 5,
    10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
};

MSL_CONSTANT const array<float, Q> wis {
    1.f / 3,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 18,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
    1.f / 36,
};
MSL_CONSTANT const array<float, Q> zerocell {
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
    0.f,
};

USERFUNC inline float feq(size_t i, float p, const vec3f& v) {
	float wi = wis[i];
	float c = cellwidth;
	float dot = offsets[i] * c * v;
	return wi * p * (1 + (1 / (c * c)) * (3 * dot + (9 / (2 * c * c)) * dot * dot - (3.f / 2) * (v * v)));
}

void zero(celerity::distr_queue& queue, celerity::buffer<cell_t , 3> buf) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class zero>(buf.get_range(), [=](celerity::item<3> item) {
			dw_buf[item] = zerocell;
		});
	});
}

void initialize(celerity::distr_queue& queue, celerity::buffer<cell_t, 3> buf, size_t size) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class initialize>(buf.get_range(), [=](celerity::item<3> item) {
			for (size_t i = 0; i < Q; i++) {
				float wi = wis[i];
				float p = 1.f;
				float c = 1.0f;
				vec3f v = {.1f, 0, 0};
				float dot = offsets[i] * c * v;
			    dw_buf[item][i] = wi * p * (1 + (1 / (c * c)) * (3 * dot + (9 / (2 * c * c)) * dot * dot - (3.f / 2) * (v * v)));
			}
			const auto xid = item.get_id(0);
			const auto yid = item.get_id(1);
			const auto zid = item.get_id(2);

			const int FLAG_OBSTACLE = 1 << 0;
			const int FLAG_KEEP_VELOCITY = 1 << 1;
			// TODO pointer results in NaN for cell[0] Kind of okay as it is not really a number but check...
			if (xid <= 1 || yid <= 1 || zid <= 1 || xid >= size - 2 || yid >= size - 2 || zid >= size - 2 ||
				std::pow(xid - 50, 2) + std::pow(yid - 50, 2) + std::pow(zid - 8, 2) <= 225) {
				auto* parts = (floatparts*) &dw_buf[item][0] ;
				parts->sign = 0;
				parts->exponent = 255;
				if (xid <= 1 || xid >= size - 2 || yid <= 1 || yid >= size - 2 || zid <= 1 || zid >= size - 2) {
					parts->mantissa = 1 << 22 | FLAG_KEEP_VELOCITY;
				} else {
					parts->mantissa = 1 << 22 | FLAG_OBSTACLE;
				}
			}
		});
	});
}
bool is_on_boundary(celerity::range<3> range, size_t filter_size, celerity::id<3> id) {
	return (id[0] < (filter_size / 2) || id[1] < (filter_size / 2) || id[2] < (filter_size / 2) ||
	        id[0] > range[0] - (filter_size / 2) - 1 || id[1] > range[1] - (filter_size / 2) - 1|| id[2] > range[2] - (filter_size / 2) - 1);
}
void update(celerity::distr_queue& queue, celerity::buffer<cell_t, 3> buf_write, celerity::buffer<cell_t, 3> buf_read, size_t size) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_buf_read{buf_read, cgh, celerity::access::neighborhood{2,2,2}, celerity::read_only};
		celerity::accessor dw_buf_write{buf_write, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class update>(celerity::range<3>(size,size,size),  [=, fs = 2](celerity::item<3> item) {
			if(is_on_boundary(celerity::range<3>(size, size, size), fs, item)) {
				dw_buf_write[item] = wis;
				return;
			}
			cell_t cell = dw_buf_read[item];
			auto* parts = (floatparts*) &cell[0];
			const int FLAG_OBSTACLE = 1 << 0;
			const int FLAG_KEEP_VELOCITY = 1 << 1;
			if (parts->exponent == 255 && parts->mantissa & FLAG_KEEP_VELOCITY) {
				dw_buf_write[item] = cell;
				return;
			}
			const auto xid = item.get_id(0);
			const auto yid = item.get_id(1);
			const auto zid = item.get_id(2);

			// Streaming.
			for (int i = 1; i < Q; i++) {
				int sx = xid + (int) offsets[i].x;
				int sy = yid + (int) offsets[i].y;
				int sz = zid + (int) offsets[i].z;
				cell_t anothercell = dw_buf_read[{xid,yid,zid}];
				cell[i] = anothercell[i];
			}

			// Collision.
			const array<unsigned char, Q> opposite = {
			    0,
			    2, 1, 4, 3, 6, 5,
			    10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15
			};
			if (parts->exponent == 255 && parts->mantissa & FLAG_OBSTACLE) {
				if (parts->mantissa & FLAG_OBSTACLE) {
					cell_t cell2 = cell;
					for (size_t i = 1; i < Q; i++) {
						cell[i] = cell2[opposite[i]];
					}
				}
				dw_buf_write[item] = cell;
				return;
			}

			float p = 0;
			vec3f vp {0, 0, 0};
			for (size_t i = 0; i < Q; i++) {
				p += cell[i];
				vp += offsets[i] * cellwidth * cell[i];
			}
			vec3f v = p == 0 ? vp : vp * (1 / p);

			for (size_t i = 0; i < Q; i++) {
				cell[i] = cell[i] + deltaT / tau * (feq(i, p, v) - cell[i]);
			}
			dw_buf_write[item] = cell; // return cell;*/
		});
	});
}

struct gassimulation_config {
	size_t nodes = 1;
	size_t threads = 64;
	size_t gpus = 1;
	size_t size = 50;
	size_t iterations = 1;
};

using arg_vector = std::vector<const char*>;

template <typename ArgFn, typename Result>
bool get_cli_arg(const arg_vector& args, const arg_vector::const_iterator& it, const std::string& argstr, Result& result, ArgFn fn) {
	if(argstr == *it) {
		if(it + 1 == args.cend()) { throw std::runtime_error("Invalid argument"); }
		result = fn(*(it + 1));
		return true;
	}
	return false;
}


int main(int argc, char* argv[]) {
	// Parse command line arguments
	const gassimulation_config cfg = ([&]() {
		gassimulation_config result;
		const arg_vector args{argv + 1, argv + argc};
		for(auto it = args.cbegin(); it != args.cend(); ++it) {
			if(get_cli_arg(args, it, "-N", result.nodes, atoi) || get_cli_arg(args, it, "-G", result.gpus, atoi)
			    || get_cli_arg(args, it, "-T", result.threads, atoi) || get_cli_arg(args, it, "--size", result.size, atoi)
			    || get_cli_arg(args, it, "--iterations", result.iterations, atoi))  {
				++it;
				continue;
			}
			std::cerr << "Unknown argument: " << *it << std::endl;
		}
		return result;
	})();

	fmt::print("Nodes={}, Threads={}, GPUs={}, x/y/z-dimension={}\n", cfg.nodes, (double)cfg.threads, (double)cfg.gpus,  (double)cfg.size);

	celerity::distr_queue queue;
	double time = MPI_Wtime();
	// Two Buffers to switch writing and empty buffer.
	celerity::buffer<cell_t, 3> actual{celerity::range<3>(cfg.size, cfg.size, cfg.size)}; // next
	celerity::buffer<cell_t, 3> swap{celerity::range<3>(cfg.size, cfg.size, cfg.size)};  // current
	// Initialize with 0.0f.
	zero(queue, swap);
	initialize(queue, actual, cfg.size);
	double timekernel = MPI_Wtime();

	for (int i = 0; i < 1; i++) {
		update(queue, swap, actual, cfg.size);
		//std::swap(actual, swap);
	}

	double end_time = MPI_Wtime();
	double totaltime = time-end_time;
	double totalkerneltime = timekernel-end_time;
	std::cout << cfg.size << ";" << cfg.iterations << ";" << cfg.nodes << ";" << cfg.threads << ";" << cfg.gpus << ";" << totaltime << ";" << totalkerneltime << std::endl;
	const celerity::experimental::host_object<std::ofstream> os;
	queue.slow_full_sync();

	FILE *file = fopen("file.txt","a+"); // apend file or create a file if it does not exist
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor actuall_buffer_all{actual, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=] {
			for(size_t x = 0; x < static_cast<size_t>(cfg.size); x++) {
			for(size_t y = 0; y < static_cast<size_t>(cfg.size); y++) {
			for(size_t z = 0; z < static_cast<size_t>(cfg.size); z++) {
				cell_t zelle = actuall_buffer_all[{x,y,z}];
				for(size_t j = 0; j < Q; j++) {
					fprintf(file, "%.4f;", zelle[j]); // write
				}
				fprintf(file, "\n"); // write
			}}}
			fprintf(file, "\n"); // write
			fclose(file);        // close file
			printf("File created. Located in the project folder.\n", "");
		});
	});


	queue.slow_full_sync();

	return EXIT_SUCCESS;
}
