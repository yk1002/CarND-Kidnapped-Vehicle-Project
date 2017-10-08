/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cassert>

#include "particle_filter.h"

using namespace std;

namespace {
void normalize(vector<double>& a) {
    const auto sum = accumulate(a.begin(), a.end(), 0.0);
    assert(sum > 0.0);
    for_each(a.begin(), a.end(), [sum](double& x){ x /= sum; });
}

LandmarkObs convertParticleFrameToMapFrame(const Particle& p, const LandmarkObs& obs) {
    LandmarkObs obs_map;
    obs_map.id = obs.id;
    obs_map.x = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
    obs_map.y = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;
    return obs_map;
}

double gaussian_pdf_2d(double x, double y, double mu_x, double mu_y, double sig_x, double sig_y) {
    assert(sig_x > 0.0);
    assert(sig_y > 0.0);

    const auto dx = x - mu_x;
    const auto dx2 = dx * dx; 
    const auto dy = y - mu_y;
    const auto dy2 = dy * dy;
    const auto sig_x2 = sig_x * sig_x;
    const auto sig_y2 = sig_y * sig_y;
    const auto exponent = -(dx2 / sig_x2 + dy2 / sig_y2) / 2.0f;

    return exp(exponent) / (2.0 * M_PI * sig_x * sig_y);
}

Map::single_landmark_s findClosestLandmark(const LandmarkObs& observation, const Map &map) {
    assert(!map.landmark_list.empty());
    
    double closest_dist = 1E60;
    int closest_index = 0;
    for (int i = 0; i < map.landmark_list.size(); ++i) {
        const auto& landmark = map.landmark_list[i];
        const auto dist_to_landmark = dist(observation.x, observation.y, landmark.x_f, landmark.y_f);
        if (dist_to_landmark < closest_dist) {
            closest_dist = dist_to_landmark;
            closest_index = i;
        }
    }

    return map.landmark_list[closest_index];
}

} // end of anon namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    if (is_initialized) return;

    normal_distribution<double> rand_x(x, std[0]), rand_y(y, std[1]), rand_theta(theta, std[2]);

    const auto weight = 1.0;
    for (int i = 0; i < num_particles; ++i) {
        particles.emplace_back(i, rand_x(gen), rand_y(gen), rand_theta(gen), weight);
        weights[i] = weight;
    }
    normalize(weights);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> rand_x(0.0, std[0]), rand_y(0.0, std[1]), rand_theta(0.0, std[2]);
    if (abs(yaw_rate) < 1E-10) {
        for (auto& p: particles) {
            p.x += velocity * delta_t * cos(p.theta) + rand_x(gen);
            p.y += velocity * delta_t * sin(p.theta) + rand_y(gen);
            p.theta += rand_theta(gen);
        }
    }
    else {
        for (auto& p: particles) {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + rand_x(gen);
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + rand_y(gen);
            p.theta += yaw_rate * delta_t + rand_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // for each particle:
    //   let score = 1.0
    //   for each observation
    //     convert the observation, which is in the particle's frame, to a map coordinate
    //     find the landmark that is closest to that map coorindate
    //     compute multivariate probability prob
    //     score = score * prob
    //   set score as the new weight for the particle
    //
    // normalize weight vector

    // TODO: How to use sensor_range?

    const auto theta_x = std_landmark[0];
    const auto theta_y = std_landmark[1];
    
    for (int i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        double score = 1.0;
        for (const auto& obs: observations) {
            // express the observation in the map frame
            const auto obs_map = convertParticleFrameToMapFrame(p, obs);
            
            // find the landmark closest to the observation
            const auto closest = findClosestLandmark(obs_map, map_landmarks);
            
            // calculate cumulative score
            score *= gaussian_pdf_2d(obs_map.x, obs_map.y, closest.x_f, closest.y_f, theta_x, theta_y);
        }

        p.weight = score;
        weights[i] = score;
    }

    normalize(weights);
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<double> dd(weights.begin(), weights.end());
    std::vector<Particle> new_particles;
    for (int i = 0; i < num_particles; ++i) {
        const auto pickedI = dd(gen);
        new_particles.emplace_back(particles[pickedI]);
    }
    swap(particles, new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
