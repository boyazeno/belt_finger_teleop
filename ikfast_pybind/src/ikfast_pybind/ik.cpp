
#include "ikfast_pybind/ikfast.h"
#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

bool satisfiesPositionBounds(const std::vector<IkReal>& solvalues, const std::vector<double>& joint_upper_limits,const std::vector<double>& joint_lower_limits, const double& margin=0.01)
{
    for (int i=0;i<solvalues.size();++i)
    {
        if (joint_upper_limits[i]-margin<solvalues[i] || joint_lower_limits[i]+margin>solvalues[i])
         return false;
    }
    return true;
}

std::tuple<bool,std::vector<double>> getIk(const Eigen::Matrix4d& target, const std::vector<double>& reference_joint_positions, const std::vector<double>& joint_upper_limits, const std::vector<double>& joint_lower_limits)
{
    Eigen::Matrix3d q_mat = target.matrix().block<3,3>(0,0);

    // Calculate ik solution
    ikfast::IkSolutionList<IkReal> solutions;
    std::vector<IkReal> vfree(GetNumFreeParameters());
    IkReal eerot[9],eetrans[3];
    IkReal joints[7];
    
    eerot[0] = q_mat.coeff(0,0); eerot[1] = q_mat.coeff(0,1); eerot[2] = q_mat.coeff(0,2); eetrans[0] = target.matrix().coeff(0,3);
    eerot[3] = q_mat.coeff(1,0); eerot[4] = q_mat.coeff(1,1); eerot[5] = q_mat.coeff(1,2); eetrans[1] = target.matrix().coeff(1,3);
    eerot[6] = q_mat.coeff(2,0); eerot[7] = q_mat.coeff(2,1); eerot[8] = q_mat.coeff(2,2); eetrans[2] = target.matrix().coeff(2,3);

    std::vector<std::vector<double>> valid_solutions;
    int discret_steps = 20; // Fixed number of free-joint samples; an adaptive strategy could improve coverage

    for (size_t i = 0; i < discret_steps; ++i)
    {
        vfree[0] = i*1.7628*2/discret_steps-1.7628;
        bool bSuccess = ComputeIk(eetrans, eerot, &vfree[0], solutions);

        if( !bSuccess ) {
            continue;
        }

        std::vector<IkReal> solvalues(GetNumJoints());
        for(std::size_t i = 0; i < solutions.GetNumSolutions(); ++i) {
            const ikfast::IkSolutionBase<IkReal>& sol = solutions.GetSolution(i);
            std::vector<IkReal> vsolfree(1);
            sol.GetSolution(&solvalues[0],&vsolfree[0]);
            // Check if valid
            if(satisfiesPositionBounds(solvalues,joint_upper_limits,joint_lower_limits, 0.01))
            {
                // Add to valid
                valid_solutions.push_back(solvalues);
            }
        }
    }

    if(valid_solutions.empty())
    {
        return {false, std::vector<double>({0,0,0,0,0,0,0})};
    }

    // Rank according to the joint distance
    const std::vector<double> weights={0.4,0.1,0.6,0.0,1.5,0.1,0.5};
    std::partial_sort(valid_solutions.begin(),valid_solutions.begin()+1,valid_solutions.end(), [&](const std::vector<double> &a, const std::vector<double> &b)
    { 
        double cost_a = 0;
        double cost_b = 0;
        for(int i=0;i<7;++i)
        {
            cost_a += std::abs(a[i]-reference_joint_positions[i])*weights[i];
            cost_b += std::abs(b[i]-reference_joint_positions[i])*weights[i];
        }
        return cost_a<cost_b; 
    });

    return {true, valid_solutions[0]};
}



PYBIND11_MODULE(ikfast_pybind, m) {
    m.doc() = "pybind11 ikfast plugin"; // optional module docstring

    m.def("get_ik", &getIk, "A function that calculate ik solution for target in link8", 
          py::arg("target"),
          py::arg("reference_joint_positions"),
          py::arg("joint_upper_limits"),
          py::arg("joint_lower_limits"));
}


// int main(int argc, char** argv)
// {
//     Eigen::Quaterniond q(0.053535, 0.298545, 0.952881, -0.004759);
//     Eigen::Matrix4d target=Eigen::Matrix4d::Identity();
//     target.matrix().block<3,3>(0,0) = q.toRotationMatrix();
//     target.matrix().block<3,1>(0,3) << 0.5, 0.5, 0.5;

//     std::vector<double> ref = {0,0,0,0,0,0,0};
//     std::vector<double> max_val = {3.14,3.14,3.14,3.14,3.14,3.14,3.14};
//     std::vector<double> min_val = {-3.14,-3.14,-3.14,-3.14,-3.14,-3.14,-3.14};
//     auto [rc, ik] = getIk(target, ref, max_val, min_val);
//     std::cout<<"get sol? "<<rc<<"\n";
//     for(auto& val : ik)
//         std::cout<<" "<<val;
//     std::cout<<std::endl;
//     return 0;
// }